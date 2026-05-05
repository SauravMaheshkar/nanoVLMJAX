import dataclasses
import json
import logging
import os
import tempfile
import types

import jax
import jax.numpy as jnp
import numpy as np
import safetensors
from jaxtyping import Array, Float

from src.models.language_model import LanguageModel, language_model_forward
from src.models.modality_projector import ModalityProjector, modality_projector_forward
from src.models.vit import ViT, vit_forward
from src.utils import ParamInitializer, ParamSpec, jax_pytree_struct

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class VLMConfig:
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 3072
    vit_patch_size: int = 16
    vit_img_size: int = 224
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False
    vit_model_type: str = "google/siglip-base-patch16-224"

    lm_hidden_dim: int = 576
    lm_inter_dim: int = 1536
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_vocab_size: int = 49152
    lm_n_heads: int = 9
    lm_n_kv_heads: int = 3
    lm_dropout: float = 0.0
    lm_n_layers: int = 30
    lm_n_blocks: int = 30  # Legacy alias for lm_n_layers
    lm_attn_scaling: float = 1.0
    lm_max_length: int = 4096
    lm_use_tokens: bool = False
    lm_tie_weights: bool = True
    lm_model_type: str = "HuggingFaceTB/SmolLM2-135M"
    lm_tokenizer: str = "HuggingFaceTB/SmolLM2-135M"
    lm_eos_token_id: int = 0
    lm_attention_bias: bool = False

    mp_pixel_shuffle_factor: int = 2
    mp_image_token_length: int = 64

    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = "nanoVLM"
    image_token_id: int = 0  # Must be set to tokenizer.image_token_id

    dtype: jnp.dtype = jnp.float32

    def __post_init__(self):
        pass


@jax_pytree_struct
class VisionLanguageModel(ParamInitializer):
    """
    Vision Language Model combining a vision encoder, modality projector,
    and language model decoder.

    References:
        * https://github.com/huggingface/nanoVLM/blob/main/models/vision_language_model.py
    """

    vision_encoder: ViT
    modality_projector: ModalityProjector
    decoder: LanguageModel
    image_token_id: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg: VLMConfig):
        vit_cfg = types.SimpleNamespace(
            vit_img_size=cfg.vit_img_size,
            vit_patch_size=cfg.vit_patch_size,
            vit_hidden_dim=cfg.vit_hidden_dim,
            vit_num_heads=cfg.vit_n_heads,
            vit_mlp_hidden_dim=cfg.vit_inter_dim,
            vit_num_blocks=cfg.vit_n_blocks,
            vit_cls_flag=cfg.vit_cls_flag,
            vit_dropout=cfg.vit_dropout,
            vit_ln_eps=cfg.vit_ln_eps,
            dtype=cfg.dtype,
            conv_weight_logical_axes=(
                "kernel_h",
                "kernel_w",
                "in_channels",
                "out_channels",
            ),
            cls_token_logical_axes=(None, "seq", "hidden"),
            position_embedding_logical_axes=(None, "seq", "hidden"),
            ln1_scale_logical_axes=("hidden",),
            ln1_bias_logical_axes=("hidden",),
            ln2_scale_logical_axes=("hidden",),
            ln2_bias_logical_axes=("hidden",),
            final_ln_scale_logical_axes=("hidden",),
            final_ln_bias_logical_axes=("hidden",),
            qkv_proj_logical_axes=("hidden", None),
            out_proj_logical_axes=("hidden", None),
            fc1_weight_logical_axes=("hidden", None),
            fc1_bias_logical_axes=(None,),
            fc2_weight_logical_axes=(None, "hidden"),
            fc2_bias_logical_axes=("hidden",),
            conv_weight_initializer=None,
            cls_token_initializer=None,
            position_embedding_initializer=None,
            ln1_scale_initializer=None,
            ln1_bias_initializer=None,
            ln2_scale_initializer=None,
            ln2_bias_initializer=None,
            final_ln_scale_initializer=None,
            final_ln_bias_initializer=None,
            qkv_proj_initializer=None,
            out_proj_initializer=None,
            fc1_weight_initializer=None,
            fc1_bias_initializer=None,
            fc2_weight_initializer=None,
            fc2_bias_initializer=None,
        )

        lm_cfg = types.SimpleNamespace(
            lm_vocab_size=cfg.lm_vocab_size,
            lm_hidden_dim=cfg.lm_hidden_dim,
            lm_n_heads=cfg.lm_n_heads,
            lm_n_kv_heads=cfg.lm_n_kv_heads,
            lm_n_layers=cfg.lm_n_layers,
            lm_inter_dim=cfg.lm_inter_dim,
            lm_dropout=cfg.lm_dropout,
            lm_rms_eps=cfg.lm_rms_eps,
            lm_re_base=float(cfg.lm_re_base),
            lm_max_position_embeddings=cfg.lm_max_position_embeddings,
            lm_attn_scaling=cfg.lm_attn_scaling,
            lm_use_tokens=cfg.lm_use_tokens,
            lm_tie_weights=cfg.lm_tie_weights,
            lm_attention_bias=cfg.lm_attention_bias,
            dtype=cfg.dtype,
            token_embedding_logical_axes=(None, "tp"),
            rotary_inv_freq_logical_axes=(None,),
            rotary_inv_freq_initializer=None,
            rms_norm_weight_logical_axes=(None,),
            rms_norm_weight_initializer=None,
            gqa_q_proj_logical_axes=("hidden", None),
            gqa_q_proj_initializer=None,
            gqa_k_proj_logical_axes=("hidden", None),
            gqa_k_proj_initializer=None,
            gqa_v_proj_logical_axes=("hidden", None),
            gqa_v_proj_initializer=None,
            gqa_out_proj_logical_axes=("hidden", None),
            gqa_out_proj_initializer=None,
            mlp_gate_proj_logical_axes=("hidden", None),
            mlp_gate_proj_initializer=None,
            mlp_up_proj_logical_axes=("hidden", None),
            mlp_up_proj_initializer=None,
            mlp_down_proj_logical_axes=(None, "hidden"),
            mlp_down_proj_initializer=None,
            block_norm1_weight_logical_axes=(None,),
            block_norm1_weight_initializer=None,
            block_norm2_weight_logical_axes=(None,),
            block_norm2_weight_initializer=None,
            final_norm_weight_logical_axes=(None,),
            final_norm_weight_initializer=None,
            head_logical_axes=("hidden", None),
            head_initializer=None,
            token_embedding_initializer=None,
        )

        mp_cfg = types.SimpleNamespace(
            vit_hidden_dim=cfg.vit_hidden_dim,
            lm_hidden_dim=cfg.lm_hidden_dim,
            mp_pixel_shuffle_factor=cfg.mp_pixel_shuffle_factor,
            dtype=cfg.dtype,
            mp_proj_logical_axes=(None, "tp"),
            mp_proj_initializer=None,
        )

        vision_encoder = ViT.param_specs(vit_cfg)
        modality_projector = ModalityProjector.param_specs(mp_cfg)
        decoder = LanguageModel.param_specs(lm_cfg)

        return VisionLanguageModel(
            vision_encoder=vision_encoder,
            modality_projector=modality_projector,
            decoder=decoder,
            image_token_id=cfg.image_token_id,
        )

    @classmethod
    def init(cls, key, mesh, rules, cfg: VLMConfig):
        return super().init(key, mesh, rules, cfg)

    @classmethod
    def from_pretrained(cls, model_id: str, *, revision: str = "main"):
        import jax.random as random

        is_local = os.path.isdir(model_id)

        if is_local:
            config_path = os.path.join(model_id, "config.json")
            safetensors_file = os.path.join(model_id, "model.safetensors")
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=model_id, filename="config.json", revision=revision
            )
            safetensors_file = hf_hub_download(
                repo_id=model_id, filename="model.safetensors", revision=revision
            )

        with open(config_path) as f:
            hf_config = json.load(f)

        # Map known PyTorch and legacy alias names to canonical VLMConfig fields.
        field_names = {f.name for f in dataclasses.fields(VLMConfig)}
        aliases = {
            "lm_base_vocab_size": "lm_vocab_size",
            "lm_n_blocks": "lm_n_layers",
        }
        unknown = []
        filtered = {}
        for k, v in hf_config.items():
            canonical = aliases.get(k, k)
            if canonical in field_names:
                filtered[canonical] = v
            else:
                unknown.append(k)
        if unknown:
            logger.warning(
                "Checkpoint config has %d unknown key(s) with no VLMConfig mapping: %s."
                "They will be ignored. If the model behaves unexpectedly, "
                "check whether VLMConfig needs updating.",
                len(unknown),
                unknown,
            )
        cfg = VLMConfig(**filtered)

        hf_state_dict = {}
        with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as f:
            import torch

            for key in f.keys():
                tensor = f.get_tensor(key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                hf_state_dict[key] = tensor.cpu().numpy()

        param_specs = cls.param_specs(cfg)

        def _init_param(param_spec):
            if param_spec.initializer is not None:
                key = random.PRNGKey(0)
                return jnp.array(
                    param_spec.initializer(key, param_spec.shape, param_spec.dtype)
                )
            return jnp.zeros(param_spec.shape, dtype=param_spec.dtype)

        def _init_pytree(spec):
            if isinstance(spec, ParamSpec):
                return _init_param(spec)
            elif hasattr(spec, "__dict__"):
                result = {}
                for k, v in spec.__dict__.items():
                    result[k] = _init_pytree(v)
                return spec.__class__(**result)
            elif isinstance(spec, list):
                return [_init_pytree(item) for item in spec]
            elif isinstance(spec, tuple):
                return tuple(_init_pytree(item) for item in spec)
            return spec

        params = _init_pytree(param_specs)

        def _map_weights(hf_dict, params):
            hf_dict = dict(hf_dict)

            vit_state = {}
            lm_state = {}
            mp_state = {}

            for key, value in hf_dict.items():
                if key.startswith("vision_encoder."):
                    vit_state[key.replace("vision_encoder.", "")] = value
                elif key.startswith("modality_projector."):
                    mp_state[key.replace("modality_projector.", "")] = value
                elif key.startswith("decoder."):
                    lm_state[key.replace("decoder.", "")] = value
                else:
                    if "proj" in key or "embedding" in key or "norm" in key:
                        lm_state[key] = value

            if "vision_encoder.patch_embedding.conv_weight" in hf_dict:
                conv_weight = jnp.array(
                    hf_dict["vision_encoder.patch_embedding.conv_weight"]
                )
                vit_state["patch_embedding.conv_weight"] = jnp.transpose(
                    conv_weight, (2, 0, 1, 3)
                )

            if "vision_encoder.patch_embedding.position_embedding" in hf_dict:
                vit_state["patch_embedding.position_embedding"] = jnp.array(
                    hf_dict["vision_encoder.patch_embedding.position_embedding"]
                )

            if "vision_encoder.layer_norm.scale" in hf_dict:
                vit_state["layer_norm.scale"] = jnp.array(
                    hf_dict["vision_encoder.layer_norm.scale"]
                )

            for i in range(cfg.vit_n_blocks):
                prefix = f"vision_encoder.blocks.{i}"

                if f"{prefix}.layer_norm1.weight" in hf_dict:
                    vit_state[f"blocks.{i}.ln1.scale"] = jnp.array(
                        hf_dict[f"{prefix}.layer_norm1.weight"]
                    )
                if f"{prefix}.layer_norm1.bias" in hf_dict:
                    vit_state[f"blocks.{i}.ln1.bias"] = jnp.array(
                        hf_dict[f"{prefix}.layer_norm1.bias"]
                    )

                if f"{prefix}.layer_norm2.weight" in hf_dict:
                    vit_state[f"blocks.{i}.ln2.scale"] = jnp.array(
                        hf_dict[f"{prefix}.layer_norm2.weight"]
                    )
                if f"{prefix}.layer_norm2.bias" in hf_dict:
                    vit_state[f"blocks.{i}.ln2.bias"] = jnp.array(
                        hf_dict[f"{prefix}.layer_norm2.bias"]
                    )

                fc1_weight = hf_dict.get(f"{prefix}.mlp.fc1.weight")
                if fc1_weight is not None:
                    vit_state[f"blocks.{i}.mlp.fc1.weight"] = jnp.array(fc1_weight).T

                fc2_weight = hf_dict.get(f"{prefix}.mlp.fc2.weight")
                if fc2_weight is not None:
                    vit_state[f"blocks.{i}.mlp.fc2.weight"] = jnp.array(fc2_weight).T

                q_w = hf_dict.get(f"{prefix}.self_attn.q_proj.weight")
                k_w = hf_dict.get(f"{prefix}.self_attn.k_proj.weight")
                v_w = hf_dict.get(f"{prefix}.self_attn.v_proj.weight")
                if q_w is not None and k_w is not None and v_w is not None:
                    vit_state[f"blocks.{i}.attn.qkv_proj.weight"] = jnp.concatenate(
                        [jnp.array(q_w), jnp.array(k_w), jnp.array(v_w)], axis=0
                    ).T

                out_proj_weight = hf_dict.get(f"{prefix}.self_attn.out_proj.weight")
                if out_proj_weight is not None:
                    vit_state[f"blocks.{i}.attn.out_proj.weight"] = jnp.array(
                        out_proj_weight
                    ).T

            if "decoder.model.embed_tokens.weight" in hf_dict:
                lm_state["token_embedding"] = jnp.array(
                    hf_dict["decoder.model.embed_tokens.weight"]
                )

            if "decoder.model.norm.weight" in hf_dict:
                lm_state["norm.weight"] = jnp.array(
                    hf_dict["decoder.model.norm.weight"]
                )

            # Handle head weights: tie to embedding if configured
            # or if no separate head is provided
            if "decoder.lm_head.weight" in hf_dict:
                lm_state["head"] = jnp.array(hf_dict["decoder.lm_head.weight"]).T
            elif "token_embedding" in lm_state and cfg.lm_tie_weights:
                lm_state["head"] = lm_state["token_embedding"].T

            for i in range(cfg.lm_n_layers):
                prefix = f"decoder.model.layers.{i}"

                if f"{prefix}.input_layernorm.weight" in hf_dict:
                    lm_state[f"blocks.{i}.norm1.weight"] = jnp.array(
                        hf_dict[f"{prefix}.input_layernorm.weight"]
                    )

                if f"{prefix}.post_attention_layernorm.weight" in hf_dict:
                    lm_state[f"blocks.{i}.norm2.weight"] = jnp.array(
                        hf_dict[f"{prefix}.post_attention_layernorm.weight"]
                    )

                gate_proj = hf_dict.get(f"{prefix}.mlp.gate_proj.weight")
                if gate_proj is not None:
                    lm_state[f"blocks.{i}.mlp.gate_proj"] = jnp.array(gate_proj).T

                up_proj = hf_dict.get(f"{prefix}.mlp.up_proj.weight")
                if up_proj is not None:
                    lm_state[f"blocks.{i}.mlp.up_proj"] = jnp.array(up_proj).T

                down_proj = hf_dict.get(f"{prefix}.mlp.down_proj.weight")
                if down_proj is not None:
                    lm_state[f"blocks.{i}.mlp.down_proj"] = jnp.array(down_proj).T

                q_proj = hf_dict.get(f"{prefix}.self_attn.q_proj.weight")
                if q_proj is not None:
                    lm_state[f"blocks.{i}.attn.q_proj"] = jnp.array(q_proj).T

                k_proj = hf_dict.get(f"{prefix}.self_attn.k_proj.weight")
                if k_proj is not None:
                    lm_state[f"blocks.{i}.attn.k_proj"] = jnp.array(k_proj).T

                v_proj = hf_dict.get(f"{prefix}.self_attn.v_proj.weight")
                if v_proj is not None:
                    lm_state[f"blocks.{i}.attn.v_proj"] = jnp.array(v_proj).T

                out_proj = hf_dict.get(f"{prefix}.self_attn.o_proj.weight")
                if out_proj is not None:
                    lm_state[f"blocks.{i}.attn.out_proj"] = jnp.array(out_proj).T

            if "modality_projector.proj.weight" in hf_dict:
                mp_state["proj"] = jnp.array(
                    hf_dict["modality_projector.proj.weight"]
                ).T

            def _set_nested_attr(obj, path, value):
                parts = path.split(".")
                for part in parts[:-1]:
                    if part.isdigit():
                        obj = obj[int(part)]
                    else:
                        obj = getattr(obj, part)
                setattr(obj, parts[-1], value)

            for jax_key, value in vit_state.items():
                try:
                    _set_nested_attr(params.vision_encoder, jax_key, value)
                except AttributeError as e:
                    logger.warning(
                        "Failed to map vision weight '%s' from checkpoint: %s",
                        jax_key,
                        e,
                    )

            for jax_key, value in lm_state.items():
                try:
                    _set_nested_attr(params.decoder, jax_key, value)
                except AttributeError as e:
                    logger.warning(
                        "Failed to map LM weight '%s' from checkpoint: %s",
                        jax_key,
                        e,
                    )

            for jax_key, value in mp_state.items():
                try:
                    _set_nested_attr(params.modality_projector, jax_key, value)
                except AttributeError as e:
                    logger.warning(
                        "Failed to map MP weight '%s' from checkpoint: %s",
                        jax_key,
                        e,
                    )

            return params

        return _map_weights(hf_state_dict, params)

    def save_pretrained(self, save_dir: str):
        import safetensors.numpy

        os.makedirs(save_dir, exist_ok=True)

        state_dict = {}

        state_dict["vision_encoder.patch_embedding.conv_weight"] = np.array(
            jnp.transpose(self.vision_encoder.patch_embedding.conv_weight, (2, 0, 1, 3))
        )
        state_dict["vision_encoder.patch_embedding.position_embedding"] = np.array(
            self.vision_encoder.patch_embedding.position_embedding
        )
        state_dict["vision_encoder.layer_norm.scale"] = np.array(
            self.vision_encoder.layer_norm.scale
        )

        for i, block in enumerate(self.vision_encoder.blocks):
            prefix = f"vision_encoder.blocks.{i}"

            state_dict[f"{prefix}.layer_norm1.weight"] = np.array(block.ln1.scale)
            state_dict[f"{prefix}.layer_norm1.bias"] = np.array(block.ln1.bias)
            state_dict[f"{prefix}.layer_norm2.weight"] = np.array(block.ln2.scale)
            state_dict[f"{prefix}.layer_norm2.bias"] = np.array(block.ln2.bias)

            if hasattr(block.mlp.fc1, "weight"):
                fc1_weight = np.array(block.mlp.fc1.weight).T
            else:
                fc1_weight = np.array(block.mlp.fc1).T
            state_dict[f"{prefix}.mlp.fc1.weight"] = fc1_weight

            if hasattr(block.mlp.fc2, "weight"):
                fc2_weight = np.array(block.mlp.fc2.weight).T
            else:
                fc2_weight = np.array(block.mlp.fc2).T
            state_dict[f"{prefix}.mlp.fc2.weight"] = fc2_weight

            if hasattr(block.attn.qkv_proj, "weight"):
                qkv_proj_array = block.attn.qkv_proj.weight
            else:
                qkv_proj_array = block.attn.qkv_proj

            qkv_weight = np.array(qkv_proj_array).T
            hidden_dim = qkv_weight.shape[0] // 3
            q_w = qkv_weight[:hidden_dim, :]
            k_w = qkv_weight[hidden_dim : 2 * hidden_dim, :]
            v_w = qkv_weight[2 * hidden_dim :, :]

            state_dict[f"{prefix}.self_attn.q_proj.weight"] = q_w
            state_dict[f"{prefix}.self_attn.k_proj.weight"] = k_w
            state_dict[f"{prefix}.self_attn.v_proj.weight"] = v_w

            if hasattr(block.attn.out_proj, "weight"):
                out_proj_weight = np.array(block.attn.out_proj.weight).T
            else:
                out_proj_weight = np.array(block.attn.out_proj).T
            state_dict[f"{prefix}.self_attn.out_proj.weight"] = out_proj_weight

        state_dict["decoder.model.embed_tokens.weight"] = np.array(
            self.decoder.token_embedding
        )
        state_dict["decoder.model.norm.weight"] = np.array(self.decoder.norm.weight)

        if self.decoder.tie_weights:
            pass
        else:
            state_dict["decoder.lm_head.weight"] = np.array(self.decoder.head).T

        for i, block in enumerate(self.decoder.blocks):
            prefix = f"decoder.model.layers.{i}"

            state_dict[f"{prefix}.input_layernorm.weight"] = np.array(
                block.norm1.weight
            )

            state_dict[f"{prefix}.post_attention_layernorm.weight"] = np.array(
                block.norm2.weight
            )

            gate_proj = np.array(block.mlp.gate_proj).T
            state_dict[f"{prefix}.mlp.gate_proj.weight"] = gate_proj

            up_proj = np.array(block.mlp.up_proj).T
            state_dict[f"{prefix}.mlp.up_proj.weight"] = up_proj

            down_proj = np.array(block.mlp.down_proj).T
            state_dict[f"{prefix}.mlp.down_proj.weight"] = down_proj

            q_proj = np.array(block.attn.q_proj).T
            state_dict[f"{prefix}.self_attn.q_proj.weight"] = q_proj

            k_proj = np.array(block.attn.k_proj).T
            state_dict[f"{prefix}.self_attn.k_proj.weight"] = k_proj

            v_proj = np.array(block.attn.v_proj).T
            state_dict[f"{prefix}.self_attn.v_proj.weight"] = v_proj

            out_proj = np.array(block.attn.out_proj).T
            state_dict[f"{prefix}.self_attn.o_proj.weight"] = out_proj

        state_dict["modality_projector.proj.weight"] = np.array(
            self.modality_projector.proj
        ).T

        safetensors.numpy.save_file(
            state_dict, os.path.join(save_dir, "model.safetensors")
        )

        config = {
            "vit_hidden_dim": self.vision_encoder.patch_embedding.embd_dim,
            "vit_inter_dim": self.vision_encoder.blocks[0].mlp.hidden_dim,
            "vit_patch_size": self.vision_encoder.patch_embedding.patch_size,
            "vit_img_size": self.vision_encoder.patch_embedding.img_size,
            "vit_n_heads": self.vision_encoder.blocks[0].attn.num_heads,
            "vit_dropout": self.vision_encoder.dropout,
            "vit_n_blocks": self.vision_encoder.num_blocks,
            "vit_ln_eps": self.vision_encoder.blocks[0].ln1.eps,
            "vit_cls_flag": self.vision_encoder.cls_flag,
            "vit_model_type": "siglip",
            "lm_hidden_dim": self.decoder.hidden_dim,
            "lm_inter_dim": self.decoder.blocks[0].mlp.inter_dim,
            "lm_rms_eps": self.decoder.blocks[0].norm1.eps,
            "lm_re_base": int(self.decoder.rotary_emb.base),
            "lm_max_position_embeddings": self.decoder.rotary_emb.max_seq_len,
            "lm_vocab_size": self.decoder.vocab_size,
            "lm_n_heads": self.decoder.blocks[0].attn.n_heads,
            "lm_n_kv_heads": self.decoder.blocks[0].attn.n_kv_heads,
            "lm_dropout": self.decoder.blocks[0].attn.dropout,
            "lm_n_layers": self.decoder.num_layers,
            "lm_attn_scaling": self.decoder.rotary_emb.attention_scaling,
            "lm_max_length": 4096,
            "lm_use_tokens": self.decoder.use_tokens,
            "lm_tie_weights": self.decoder.tie_weights,
            "lm_model_type": "SmolLM",
            "image_token_id": self.image_token_id,
            "lm_tokenizer": "HuggingFaceTB/SmolLM2-135M",
            "lm_eos_token_id": 0,
            "mp_pixel_shuffle_factor": self.modality_projector.scale_factor,
            "vlm_load_backbone_weights": True,
            "vlm_checkpoint_path": "nanoVLM",
        }

        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def push_to_hub(
        self,
        model_id: str,
        *,
        private: bool = False,
        commit_message: str = "Upload VLM model",
    ):
        from huggingface_hub import HfApi, create_repo

        with tempfile.TemporaryDirectory() as tmpdir:
            self.save_pretrained(tmpdir)

            create_repo(model_id, private=private, exist_ok=True)

            api = HfApi()
            api.upload_folder(
                folder_path=tmpdir,
                repo_id=model_id,
                repo_type="model",
                commit_message=commit_message,
            )


def vlm_forward(
    params: VisionLanguageModel,
    input_ids: Float[Array, "B S"],
    images: Float[Array, "B C H W"] | None = None,
    key: jax.Array | None = None,
    attention_mask: Float[Array, "B S"] | None = None,
) -> Float[Array, "B S V"]:
    """
    Forward pass through the Vision Language Model.

    Args:
        params: VLM parameters
        input_ids: Input token IDs of shape (B, S)
        images: Input images of shape (B, C, H, W) or None
        key: Optional PRNG key for dropout
        attention_mask: Optional attention mask

    Returns:
        Output logits of shape (B, S, V)
    """
    token_embd = params.decoder.token_embedding[input_ids]

    if images is not None:
        image_embd = vit_forward(params.vision_encoder, images, key)
        image_embd = modality_projector_forward(params.modality_projector, image_embd)

        image_mask = input_ids == params.image_token_id
        num_image_tokens = image_embd.shape[1]
        seq_len = token_embd.shape[1]

        padded_image_embd = jnp.pad(
            image_embd,
            ((0, 0), (0, seq_len - num_image_tokens), (0, 0)),
        )[:, :seq_len, :]

        token_embd = jnp.where(
            image_mask[:, :, None],
            padded_image_embd,
            token_embd,
        )

    logits = language_model_forward(
        params.decoder, input_ids, key, attention_mask, token_embd=token_embd
    )

    return logits
