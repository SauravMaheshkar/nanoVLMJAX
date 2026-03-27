import dataclasses
import math
import types
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from src.utils import ParamInitializer, ParamSpec, jax_pytree_struct

_embed_init = jax.nn.initializers.uniform(scale=1)


def linear_init(fan_in, fan_out) -> Callable:
    std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
    return jax.nn.initializers.normal(stddev=std)


#### RMSNorm ####


@jax_pytree_struct
class RMSNorm(ParamInitializer):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Normalizes the input across the last dimension using RMS normalization,
    which scales the input without subtracting the mean. Commonly used as a
    lighter alternative to LayerNorm in transformer models.

    References:
        * https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L69
    """

    weight: jax.Array | ParamSpec
    normalized_shape: int = dataclasses.field(metadata=dict(static=True))
    eps: float = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        weight = ParamSpec(
            shape=(cfg.lm_hidden_dim,),
            dtype=cfg.dtype,
            logical_axes=cfg.rms_norm_weight_logical_axes,
            initializer=cfg.rms_norm_weight_initializer or jax.nn.initializers.ones,
        )

        return RMSNorm(
            weight=weight,
            normalized_shape=cfg.lm_hidden_dim,
            eps=cfg.lm_rms_eps,
        )

    @classmethod
    def init(cls, key, mesh, rules, cfg):
        return super().init(key, mesh, rules, cfg)


def rms_norm_forward(
    params: RMSNorm,
    x: Float[Array, "... D"],
) -> Float[Array, "... D"]:
    """
    Forward pass for RMSNorm.

    Args:
        params: RMSNorm parameters
        x: Input tensor of shape (..., lm_hidden_dim)

    Returns:
        Normalized tensor of the same shape as input.
    """
    irms = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + params.eps)
    return x * irms * params.weight


#### Rotary Embedding ####


def rotate_half(x: Float[Array, "... D"]) -> Float[Array, "... D"]:
    """
    Rotates the input by dividing the hidden dimension in half, then swapping
    and negating the second half.
    """
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: Float[Array, "B H S D"],
    k: Float[Array, "B H S D"],
    cos: Float[Array, "B S D"],
    sin: Float[Array, "B S D"],
) -> tuple[Float[Array, "B H S D"], Float[Array, "B H S D"]]:
    """
    Applies rotary positional embeddings to query and key tensors.

    Args:
        q: Query tensor with shape (B, H, S, D)
        k: Key tensor with shape (B, H, S, D)
        cos: Cosine positional embeddings with shape (B, S, D)
        sin: Sine positional embeddings with shape (B, S, D)

    Returns:
        Tuple of rotated query and key tensors.
    """
    cos = jnp.expand_dims(cos, axis=1)
    sin = jnp.expand_dims(sin, axis=1)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


@jax_pytree_struct
class RotaryEmbedding(ParamInitializer):
    """
    Computes Rotary Position Embedding to introduce positional dependency
    to the input sequence without additional training parameters.

    References:
        * https://github.com/huggingface/nanoVLM/blob/main/models/language_model.py
    """

    inv_freq: jax.Array | ParamSpec
    head_dim: int = dataclasses.field(metadata=dict(static=True))
    base: float = dataclasses.field(metadata=dict(static=True))
    max_seq_len: int = dataclasses.field(metadata=dict(static=True))
    attention_scaling: float = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        head_dim = cfg.lm_hidden_dim // cfg.lm_n_heads
        inv_freq_value = 1.0 / (
            cfg.lm_re_base ** (jnp.arange(0, head_dim, 2) / head_dim)
        )

        if cfg.rotary_inv_freq_initializer:
            inv_freq_initializer = cfg.rotary_inv_freq_initializer
        else:

            def inv_freq_initializer(key, shape, dtype):
                value = inv_freq_value.astype(dtype)
                return jnp.broadcast_to(value, shape)

        inv_freq = ParamSpec(
            shape=inv_freq_value.shape,
            dtype=cfg.dtype,
            logical_axes=cfg.rotary_inv_freq_logical_axes,
            initializer=inv_freq_initializer,
        )

        return RotaryEmbedding(
            inv_freq=inv_freq,
            head_dim=head_dim,
            base=cfg.lm_re_base,
            max_seq_len=cfg.lm_max_position_embeddings,
            attention_scaling=cfg.lm_attn_scaling,
        )

    @classmethod
    def init(cls, key, mesh, rules, cfg):
        return super().init(key, mesh, rules, cfg)


def rotary_embedding_forward(
    params: RotaryEmbedding,
    position_ids: Float[Array, "B S"],
) -> tuple[Float[Array, "B S D"], Float[Array, "B S D"]]:
    """
    Compute rotary positional embeddings (cosine and sine components).

    Args:
        params: RotaryEmbedding parameters
        position_ids: Position indices tensor of shape (B, S)

    Returns:
        Tuple of (cos, sin) tensors, each of shape (B, S, D)
    """
    B, S = position_ids.shape

    flat_position_ids = position_ids.reshape(-1).astype(jnp.float32)
    freqs = flat_position_ids[:, None] * params.inv_freq[None, :]
    freqs = freqs.reshape(B, S, -1)

    emb = jnp.concatenate([freqs, freqs], axis=-1)
    cos = jnp.cos(emb) * params.attention_scaling
    sin = jnp.sin(emb) * params.attention_scaling

    return cos, sin


#### Grouped Query Attention ####


@jax_pytree_struct
class LanguageModelGroupedQueryAttention(ParamInitializer):
    """
    Implements Grouped Query Attention (GQA) as used in transformer-based
    language models. GQA reduces computation by using fewer key-value heads
    than query heads.

    References:
        * https://github.com/huggingface/nanoVLM/blob/main/models/language_model.py
    """

    q_proj: jax.Array | ParamSpec
    k_proj: jax.Array | ParamSpec
    v_proj: jax.Array | ParamSpec
    out_proj: jax.Array | ParamSpec
    n_heads: int = dataclasses.field(metadata=dict(static=True))
    n_kv_heads: int = dataclasses.field(metadata=dict(static=True))
    head_dim: int = dataclasses.field(metadata=dict(static=True))
    dropout: float = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        assert cfg.lm_n_heads % cfg.lm_n_kv_heads == 0, (
            "n_heads must be divisible by n_kv_heads"
        )
        assert cfg.lm_hidden_dim % cfg.lm_n_heads == 0, (
            "embd_dim must be divisible by num_heads"
        )

        head_dim = cfg.lm_hidden_dim // cfg.lm_n_heads

        q_proj = ParamSpec(
            shape=(cfg.lm_hidden_dim, cfg.lm_hidden_dim),
            dtype=cfg.dtype,
            logical_axes=cfg.gqa_q_proj_logical_axes,
            initializer=cfg.gqa_q_proj_initializer
            or linear_init(cfg.lm_hidden_dim, cfg.lm_hidden_dim),
        )

        k_proj = ParamSpec(
            shape=(cfg.lm_hidden_dim, head_dim * cfg.lm_n_kv_heads),
            dtype=cfg.dtype,
            logical_axes=cfg.gqa_k_proj_logical_axes,
            initializer=cfg.gqa_k_proj_initializer
            or linear_init(cfg.lm_hidden_dim, head_dim * cfg.lm_n_kv_heads),
        )

        v_proj = ParamSpec(
            shape=(cfg.lm_hidden_dim, head_dim * cfg.lm_n_kv_heads),
            dtype=cfg.dtype,
            logical_axes=cfg.gqa_v_proj_logical_axes,
            initializer=cfg.gqa_v_proj_initializer
            or linear_init(cfg.lm_hidden_dim, head_dim * cfg.lm_n_kv_heads),
        )

        out_proj = ParamSpec(
            shape=(cfg.lm_hidden_dim, cfg.lm_hidden_dim),
            dtype=cfg.dtype,
            logical_axes=cfg.gqa_out_proj_logical_axes,
            initializer=cfg.gqa_out_proj_initializer
            or linear_init(cfg.lm_hidden_dim, cfg.lm_hidden_dim),
        )

        return LanguageModelGroupedQueryAttention(
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            out_proj=out_proj,
            n_heads=cfg.lm_n_heads,
            n_kv_heads=cfg.lm_n_kv_heads,
            head_dim=head_dim,
            dropout=cfg.lm_dropout,
        )

    @classmethod
    def init(cls, key, mesh, rules, cfg):
        return super().init(key, mesh, rules, cfg)


def language_model_gqa_forward(
    params: LanguageModelGroupedQueryAttention,
    x: Float[Array, "B S D"],
    cos: Float[Array, "B S D"],
    sin: Float[Array, "B S D"],
    key: jax.Array | None = None,
    attention_mask: Float[Array, "B S"] | None = None,
) -> Float[Array, "B S D"]:
    """
    Forward pass for Grouped Query Attention.

    Args:
        params: GQA parameters
        x: Input tensor of shape (B, S, D)
        cos: Rotary embedding cosines
        sin: Rotary embedding sines
        key: Optional PRNG key for dropout
        attention_mask: Optional attention mask of shape (B, S)

    Returns:
        Output tensor of shape (B, S, D)
    """
    B, S, D = x.shape
    n_kv_groups = params.n_heads // params.n_kv_heads

    with jax.named_scope("q_proj"):
        q = jnp.dot(x, params.q_proj).reshape(B, S, params.n_heads, params.head_dim)
        q = q.transpose(0, 2, 1, 3)

    with jax.named_scope("k_proj"):
        k = jnp.dot(x, params.k_proj).reshape(B, S, params.n_kv_heads, params.head_dim)
        k = k.transpose(0, 2, 1, 3)

    with jax.named_scope("v_proj"):
        v = jnp.dot(x, params.v_proj).reshape(B, S, params.n_kv_heads, params.head_dim)
        v = v.transpose(0, 2, 1, 3)

    with jax.named_scope("rotary_emb"):
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

    with jax.named_scope("repeat_kv"):
        k = jnp.repeat(k, n_kv_groups, axis=1)
        v = jnp.repeat(v, n_kv_groups, axis=1)

    with jax.named_scope("attention"):
        scale = 1.0 / math.sqrt(params.head_dim)
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

        causal_mask = jnp.tril(jnp.ones((S, S), dtype=bool))[None, None, :, :]

        if attention_mask is not None:
            padding_mask = attention_mask[:, None, None, :].astype(bool)
            combined_mask = jnp.logical_and(causal_mask, padding_mask)
        else:
            combined_mask = causal_mask

        attn = jnp.where(combined_mask, attn, -1e9)

        attn = jax.nn.softmax(attn, axis=-1)

        if key is not None:
            attn_key, resid_key = jax.random.split(key)
            keep_prob = 1.0 - params.dropout
            dropout_mask = jax.random.bernoulli(attn_key, keep_prob, attn.shape)
            attn = attn * dropout_mask / keep_prob
        else:
            resid_key = None

        y = jnp.einsum("bhqk,bhvd->bhqd", attn, v)

    with jax.named_scope("out_proj"):
        y = y.transpose(0, 2, 1, 3).reshape(B, S, D)
        y = jnp.dot(y, params.out_proj)

    with jax.named_scope("resid_dropout"):
        if resid_key is not None:
            keep_prob = 1.0 - params.dropout
            dropout_mask = jax.random.bernoulli(resid_key, keep_prob, y.shape)
            y = y * dropout_mask / keep_prob

    return y


#### MLP ####


@jax_pytree_struct
class LanguageModelMLP(ParamInitializer):
    """
    Implements the feed-forward network (MLP) block used in transformer-based
    language models. This MLP uses a gated activation mechanism (SwiGLU).

    References:
        * https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L160
    """

    gate_proj: jax.Array | ParamSpec
    up_proj: jax.Array | ParamSpec
    down_proj: jax.Array | ParamSpec
    hidden_dim: int = dataclasses.field(metadata=dict(static=True))
    inter_dim: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        gate_proj = ParamSpec(
            shape=(cfg.lm_hidden_dim, cfg.lm_inter_dim),
            dtype=cfg.dtype,
            logical_axes=cfg.mlp_gate_proj_logical_axes,
            initializer=cfg.mlp_gate_proj_initializer
            or linear_init(cfg.lm_hidden_dim, cfg.lm_inter_dim),
        )

        up_proj = ParamSpec(
            shape=(cfg.lm_hidden_dim, cfg.lm_inter_dim),
            dtype=cfg.dtype,
            logical_axes=cfg.mlp_up_proj_logical_axes,
            initializer=cfg.mlp_up_proj_initializer
            or linear_init(cfg.lm_hidden_dim, cfg.lm_inter_dim),
        )

        down_proj = ParamSpec(
            shape=(cfg.lm_inter_dim, cfg.lm_hidden_dim),
            dtype=cfg.dtype,
            logical_axes=cfg.mlp_down_proj_logical_axes,
            initializer=cfg.mlp_down_proj_initializer
            or linear_init(cfg.lm_inter_dim, cfg.lm_hidden_dim),
        )

        return LanguageModelMLP(
            gate_proj=gate_proj,
            up_proj=up_proj,
            down_proj=down_proj,
            hidden_dim=cfg.lm_hidden_dim,
            inter_dim=cfg.lm_inter_dim,
        )

    @classmethod
    def init(cls, key, mesh, rules, cfg):
        return super().init(key, mesh, rules, cfg)


def language_model_mlp_forward(
    params: LanguageModelMLP,
    x: Float[Array, "B S D"],
) -> Float[Array, "B S D"]:
    """
    Forward pass through the gated MLP block (SwiGLU).

    Args:
        params: MLP parameters
        x: Input tensor of shape (B, S, D)

    Returns:
        Output tensor of shape (B, S, D)
    """
    with jax.named_scope("gate_proj"):
        gate = jax.nn.silu(jnp.dot(x, params.gate_proj))

    with jax.named_scope("up_proj"):
        up = jnp.dot(x, params.up_proj)

    with jax.named_scope("down_proj"):
        return jnp.dot(gate * up, params.down_proj)


#### Language Model Block ####


@jax_pytree_struct
class LanguageModelBlock(ParamInitializer):
    """
    A single transformer block for the language model.

    References:
        * https://github.com/meta-llama/llama3/blob/main/llama/model.py#L222
    """

    norm1: RMSNorm
    attn: LanguageModelGroupedQueryAttention
    norm2: RMSNorm
    mlp: LanguageModelMLP

    @classmethod
    def param_specs(cls, cfg):
        norm1 = RMSNorm.param_specs(
            types.SimpleNamespace(
                lm_hidden_dim=cfg.lm_hidden_dim,
                lm_rms_eps=cfg.lm_rms_eps,
                dtype=cfg.dtype,
                rms_norm_weight_logical_axes=cfg.block_norm1_weight_logical_axes,
                rms_norm_weight_initializer=cfg.block_norm1_weight_initializer,
            )
        )

        norm2 = RMSNorm.param_specs(
            types.SimpleNamespace(
                lm_hidden_dim=cfg.lm_hidden_dim,
                lm_rms_eps=cfg.lm_rms_eps,
                dtype=cfg.dtype,
                rms_norm_weight_logical_axes=cfg.block_norm2_weight_logical_axes,
                rms_norm_weight_initializer=cfg.block_norm2_weight_initializer,
            )
        )

        attn = LanguageModelGroupedQueryAttention.param_specs(cfg)
        mlp = LanguageModelMLP.param_specs(cfg)

        return LanguageModelBlock(
            norm1=norm1,
            attn=attn,
            norm2=norm2,
            mlp=mlp,
        )

    @classmethod
    def init(cls, key, mesh, rules, cfg):
        return super().init(key, mesh, rules, cfg)


def language_model_block_forward(
    params: LanguageModelBlock,
    x: Float[Array, "B S D"],
    cos: Float[Array, "B S D"],
    sin: Float[Array, "B S D"],
    key: jax.Array | None = None,
    attention_mask: Float[Array, "B S"] | None = None,
) -> Float[Array, "B S D"]:
    """
    Forward pass through a single transformer block.

    Args:
        params: Block parameters
        x: Input tensor of shape (B, S, D)
        cos: Rotary embedding cosines
        sin: Rotary embedding sines
        key: Optional PRNG key for dropout
        attention_mask: Optional attention mask

    Returns:
        Output tensor of shape (B, S, D)
    """
    if key is not None:
        attn_key, _ = jax.random.split(key)
    else:
        attn_key = None

    with jax.named_scope("norm1"):
        residual = x
        x = rms_norm_forward(params.norm1, x)

    with jax.named_scope("attn"):
        x = language_model_gqa_forward(
            params.attn, x, cos, sin, attn_key, attention_mask
        )
        x = x + residual

    with jax.named_scope("norm2"):
        residual = x
        x = rms_norm_forward(params.norm2, x)

    with jax.named_scope("mlp"):
        x = language_model_mlp_forward(params.mlp, x)
        x = x + residual

    return x


#### Language Model ####


@jax_pytree_struct
class LanguageModel(ParamInitializer):
    """
    Full Language Model consisting of token embeddings, rotary embeddings,
    multiple transformer blocks, and a final output projection.

    References:
        * https://github.com/huggingface/nanoVLM/blob/main/models/language_model.py
    """

    token_embedding: jax.Array | ParamSpec
    rotary_emb: RotaryEmbedding
    blocks: list
    norm: RMSNorm
    head: jax.Array | ParamSpec
    vocab_size: int = dataclasses.field(metadata=dict(static=True))
    hidden_dim: int = dataclasses.field(metadata=dict(static=True))
    num_layers: int = dataclasses.field(metadata=dict(static=True))
    use_tokens: bool = dataclasses.field(metadata=dict(static=True))
    tie_weights: bool = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        token_embedding = ParamSpec(
            shape=(cfg.lm_vocab_size, cfg.lm_hidden_dim),
            dtype=cfg.dtype,
            logical_axes=cfg.token_embedding_logical_axes,
            initializer=cfg.token_embedding_initializer or _embed_init,
        )

        rotary_emb = RotaryEmbedding.param_specs(cfg)

        blocks = [LanguageModelBlock.param_specs(cfg) for _ in range(cfg.lm_n_layers)]

        norm = RMSNorm.param_specs(
            types.SimpleNamespace(
                lm_hidden_dim=cfg.lm_hidden_dim,
                lm_rms_eps=cfg.lm_rms_eps,
                dtype=cfg.dtype,
                rms_norm_weight_logical_axes=cfg.final_norm_weight_logical_axes,
                rms_norm_weight_initializer=cfg.final_norm_weight_initializer,
            )
        )

        head = ParamSpec(
            shape=(cfg.lm_hidden_dim, cfg.lm_vocab_size),
            dtype=cfg.dtype,
            logical_axes=cfg.head_logical_axes,
            initializer=cfg.head_initializer
            or linear_init(cfg.lm_hidden_dim, cfg.lm_vocab_size),
        )

        return LanguageModel(
            token_embedding=token_embedding,
            rotary_emb=rotary_emb,
            blocks=blocks,
            norm=norm,
            head=head,
            vocab_size=cfg.lm_vocab_size,
            hidden_dim=cfg.lm_hidden_dim,
            num_layers=cfg.lm_n_layers,
            use_tokens=cfg.lm_use_tokens,
            tie_weights=cfg.lm_tie_weights,
        )

    @classmethod
    def init(cls, key, mesh, rules, cfg):
        return super().init(key, mesh, rules, cfg)

    @classmethod
    def from_pretrained(cls, model_id: str, *, revision: str = "main"):
        import json
        import logging
        import os
        import types

        import jax.random as random
        import safetensors
        import torch
        from huggingface_hub import hf_hub_download

        logger = logging.getLogger(__name__)

        is_local = os.path.isdir(model_id)

        if is_local:
            config_path = os.path.join(model_id, "config.json")
            safetensors_file = os.path.join(model_id, "model.safetensors")
        else:
            config_path = hf_hub_download(
                repo_id=model_id, filename="config.json", revision=revision
            )
            safetensors_file = hf_hub_download(
                repo_id=model_id, filename="model.safetensors", revision=revision
            )

        with open(config_path) as f:
            hf_config = json.load(f)

        cfg = types.SimpleNamespace(
            lm_model_type=model_id,
            lm_vocab_size=hf_config.get("vocab_size", 49152),
            lm_hidden_dim=hf_config.get("hidden_size", 576),
            lm_n_heads=hf_config.get("num_attention_heads", 9),
            lm_n_kv_heads=hf_config.get("num_key_value_heads", 3),
            lm_n_layers=hf_config.get("num_hidden_layers", 30),
            lm_inter_dim=hf_config.get("intermediate_size", 1536),
            lm_dropout=hf_config.get("attention_dropout", 0.0),
            lm_rms_eps=hf_config.get("rms_norm_eps", 1e-6),
            lm_re_base=hf_config.get("rope_theta", 100000.0),
            lm_max_position_embeddings=hf_config.get("max_position_embeddings", 8192),
            lm_attn_scaling=1.0,
            lm_use_tokens=True,
            lm_tie_weights=hf_config.get("tie_word_embeddings", True),
            lm_attention_bias=hf_config.get("attention_bias", False),
            dtype=jnp.float32,
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

        hf_state_dict = {}
        with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                hf_state_dict[key] = tensor

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

            hf_dict["token_embedding"] = jnp.array(
                hf_dict.pop("model.embed_tokens.weight")
            )

            hf_dict["norm.weight"] = jnp.array(hf_dict.pop("model.norm.weight"))

            if cfg.lm_tie_weights:
                hf_dict["head"] = hf_dict["token_embedding"].T
            else:
                hf_dict["head"] = jnp.array(hf_dict.pop("lm_head.weight")).T

            for i in range(cfg.lm_n_layers):
                prefix = f"model.layers.{i}"

                hf_dict[f"blocks.{i}.norm1.weight"] = jnp.array(
                    hf_dict.pop(f"{prefix}.input_layernorm.weight")
                )

                hf_dict[f"blocks.{i}.norm2.weight"] = jnp.array(
                    hf_dict.pop(f"{prefix}.post_attention_layernorm.weight")
                )

                gate_proj = jnp.array(hf_dict.pop(f"{prefix}.mlp.gate_proj.weight"))
                hf_dict[f"blocks.{i}.mlp.gate_proj"] = gate_proj.T

                up_proj = jnp.array(hf_dict.pop(f"{prefix}.mlp.up_proj.weight"))
                hf_dict[f"blocks.{i}.mlp.up_proj"] = up_proj.T

                down_proj = jnp.array(hf_dict.pop(f"{prefix}.mlp.down_proj.weight"))
                hf_dict[f"blocks.{i}.mlp.down_proj"] = down_proj.T

                q_proj = jnp.array(hf_dict.pop(f"{prefix}.self_attn.q_proj.weight"))
                hf_dict[f"blocks.{i}.attn.q_proj"] = q_proj.T

                k_proj = jnp.array(hf_dict.pop(f"{prefix}.self_attn.k_proj.weight"))
                hf_dict[f"blocks.{i}.attn.k_proj"] = k_proj.T

                v_proj = jnp.array(hf_dict.pop(f"{prefix}.self_attn.v_proj.weight"))
                hf_dict[f"blocks.{i}.attn.v_proj"] = v_proj.T

                out_proj = jnp.array(hf_dict.pop(f"{prefix}.self_attn.o_proj.weight"))
                hf_dict[f"blocks.{i}.attn.out_proj"] = out_proj.T

            def _set_nested_attr(obj, path, value):
                parts = path.split(".")
                i = 0
                while i < len(parts) - 1:
                    part = parts[i]
                    if part == "blocks":
                        i += 1
                        idx = int(parts[i])
                        obj = obj.blocks[idx]
                    elif part == "rotary_emb":
                        obj = obj.rotary_emb
                    else:
                        obj = getattr(obj, part)
                    i += 1
                setattr(obj, parts[-1], value)

            for jax_key, value in hf_dict.items():
                if jax_key in ["token_embedding", "head", "norm.weight"]:
                    try:
                        _set_nested_attr(params, jax_key, value)
                    except AttributeError as e:
                        logger.warning(
                            "Failed to map weight '%s' from checkpoint: %s",
                            jax_key,
                            e,
                        )
                elif jax_key.startswith("blocks.") or jax_key.startswith("norm."):
                    try:
                        _set_nested_attr(params, jax_key, value)
                    except AttributeError as e:
                        logger.warning(
                            "Failed to map weight '%s' from checkpoint: %s",
                            jax_key,
                            e,
                        )

            head_dim = cfg.lm_hidden_dim // cfg.lm_n_heads
            inv_freq = 1.0 / (cfg.lm_re_base ** (jnp.arange(0, head_dim, 2) / head_dim))
            params.rotary_emb.inv_freq = inv_freq

            return params

        return _map_weights(hf_state_dict, params)

    def save_pretrained(self, save_dir: str):
        import json
        import os

        import numpy as np
        import safetensors.numpy

        os.makedirs(save_dir, exist_ok=True)

        state_dict = {}

        state_dict["model.embed_tokens.weight"] = np.array(self.token_embedding)

        state_dict["model.norm.weight"] = np.array(self.norm.weight)

        if self.tie_weights:
            pass
        else:
            state_dict["lm_head.weight"] = np.array(self.head).T

        for i, block in enumerate(self.blocks):
            prefix = f"model.layers.{i}"

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

        safetensors.numpy.save_file(
            state_dict, os.path.join(save_dir, "model.safetensors")
        )

        config = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_dim,
            "num_attention_heads": self.blocks[0].attn.n_heads,
            "num_key_value_heads": self.blocks[0].attn.n_kv_heads,
            "num_hidden_layers": self.num_layers,
            "intermediate_size": self.blocks[0].mlp.inter_dim,
            "rms_norm_eps": self.blocks[0].norm1.eps,
            "rope_theta": self.rotary_emb.base,
            "max_position_embeddings": self.rotary_emb.max_seq_len,
            "tie_word_embeddings": self.tie_weights,
            "attention_dropout": 0.0,
        }

        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def push_to_hub(
        self,
        model_id: str,
        *,
        private: bool = False,
        commit_message: str = "Upload Language Model",
    ) -> None:
        import tempfile

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


def language_model_forward(
    params: LanguageModel,
    input_ids: Int[Array, "B S"],
    key: jax.Array | None = None,
    attention_mask: Float[Array, "B S"] | None = None,
    token_embd: Float[Array, "B S D"] | None = None,
) -> Float[Array, "B S V"]:
    """
    Forward pass through the language model.

    Args:
        params: Language model parameters
        input_ids: Input token IDs of shape (B, S)
        key: Optional PRNG key for dropout
        attention_mask: Optional attention mask of shape (B, S)
        token_embd: Optional pre-computed token embeddings of shape (B, S, D).
                   If provided, skips the embedding lookup.

    Returns:
        Output logits of shape (B, S, V)
    """
    with jax.named_scope("token_embedding"):
        if token_embd is None:
            x = params.token_embedding[input_ids]
        else:
            x = token_embd

    with jax.named_scope("rotary_emb"):
        B, S = input_ids.shape
        position_ids = jnp.broadcast_to(jnp.arange(S)[None, :], (B, S))
        cos, sin = rotary_embedding_forward(params.rotary_emb, position_ids)

    keys = (
        jax.random.split(key, params.num_layers)
        if key is not None
        else [None] * params.num_layers
    )

    with jax.named_scope("blocks"):
        for i, block in enumerate(params.blocks):
            x = language_model_block_forward(
                block, x, cos, sin, keys[i], attention_mask
            )

    with jax.named_scope("norm"):
        x = rms_norm_forward(params.norm, x)

    with jax.named_scope("head"):
        x = jnp.dot(x, params.head)

    return x
