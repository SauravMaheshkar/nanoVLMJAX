import dataclasses
import logging
import math
import types
from functools import lru_cache
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from src.utils import ParamInitializer, ParamSpec, jax_pytree_struct

logger = logging.getLogger(__name__)

_he_normal = lru_cache(jax.nn.initializers.he_normal)
_embed_init = jax.nn.initializers.uniform(scale=1)


def linear_init(fan_in, fan_out) -> Callable:
    std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
    return jax.nn.initializers.normal(stddev=std)


#### LayerNorm ####


@jax_pytree_struct
class LayerNorm(ParamInitializer):
    scale: jax.Array | ParamSpec
    bias: jax.Array | ParamSpec
    normalized_shape: int = dataclasses.field(metadata=dict(static=True))
    eps: float = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        scale = ParamSpec(
            shape=(cfg.normalized_shape,),
            dtype=cfg.dtype,
            logical_axes=cfg.scale_logical_axes,
            initializer=cfg.scale_initializer or jax.nn.initializers.ones,
        )

        bias = ParamSpec(
            shape=(cfg.normalized_shape,),
            dtype=cfg.dtype,
            logical_axes=cfg.bias_logical_axes,
            initializer=cfg.bias_initializer or jax.nn.initializers.zeros,
        )

        return LayerNorm(
            scale=scale,
            bias=bias,
            normalized_shape=cfg.normalized_shape,
            eps=cfg.eps,
        )

    @classmethod
    def init(cls, key, mesh, rules, cfg):
        return super().init(key, mesh, rules, cfg)


def layer_norm_forward(
    params: LayerNorm,
    x: Float[Array, "... D"],
) -> Float[Array, "... D"]:
    mean = jnp.mean(x, axis=-1, keepdims=True)  # (..., 1)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)  # (..., 1)
    x_norm = (x - mean) / jnp.sqrt(variance + params.eps)  # (..., D)
    return params.scale * x_norm + params.bias  # (..., D)


#### Linear Layer ####


@jax_pytree_struct
class Linear(ParamInitializer):
    """
    References:
    * https://github.com/AakashKumarNain/nanoGPTJAX/blob/main/nanogpt/layers.py
    """

    weight: jax.Array | ParamSpec
    bias: jax.Array | ParamSpec | None
    in_features: int = dataclasses.field(metadata=dict(static=True))
    out_features: int = dataclasses.field(metadata=dict(static=True))
    use_bias: bool = dataclasses.field(default=True, metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        weight = ParamSpec(
            shape=(cfg.in_features, cfg.out_features),
            dtype=cfg.dtype,
            logical_axes=cfg.weight_logical_axes,
            initializer=cfg.weight_initializer
            or linear_init(cfg.in_features, cfg.out_features),
        )

        if cfg.use_bias:
            bias = ParamSpec(
                shape=(cfg.out_features,),
                dtype=cfg.dtype,
                logical_axes=cfg.bias_logical_axes,
                initializer=cfg.bias_initializer or jax.nn.initializers.zeros,
            )
        else:
            bias = None

        return Linear(
            weight=weight,
            bias=bias,
            in_features=cfg.in_features,
            out_features=cfg.out_features,
            use_bias=cfg.use_bias,
        )

    @classmethod
    def init(cls, key, mesh, rules, cfg):
        return super().init(key, mesh, rules, cfg)


def linear_forward(
    params: Linear,
    x: Float[Array, "... D"],
) -> Float[Array, "... O"]:
    """
    References:
    * https://github.com/AakashKumarNain/nanoGPTJAX/blob/main/nanogpt/model.py
    """
    out = jnp.einsum("...d,do->...o", x, params.weight)  # (..., out_features)
    if params.bias is not None:
        out = out + params.bias  # (..., out_features)
    return out


#### MLP ####


@jax_pytree_struct
class ViTMLP(ParamInitializer):
    """
    References:
    * https://github.com/AakashKumarNain/nanoGPTJAX/blob/main/nanogpt/model.py
    """

    fc1: Linear
    fc2: Linear
    hidden_dim: int = dataclasses.field(metadata=dict(static=True))
    mlp_hidden_dim: int = dataclasses.field(metadata=dict(static=True))
    dropout: float = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        fc1 = Linear.param_specs(
            types.SimpleNamespace(
                in_features=cfg.vit_hidden_dim,
                out_features=cfg.vit_mlp_hidden_dim,
                dtype=cfg.dtype,
                weight_logical_axes=cfg.fc1_weight_logical_axes,
                bias_logical_axes=cfg.fc1_bias_logical_axes,
                weight_initializer=cfg.fc1_weight_initializer,
                bias_initializer=cfg.fc1_bias_initializer,
                use_bias=True,
            )
        )

        fc2 = Linear.param_specs(
            types.SimpleNamespace(
                in_features=cfg.vit_mlp_hidden_dim,
                out_features=cfg.vit_hidden_dim,
                dtype=cfg.dtype,
                weight_logical_axes=cfg.fc2_weight_logical_axes,
                bias_logical_axes=cfg.fc2_bias_logical_axes,
                weight_initializer=cfg.fc2_weight_initializer,
                bias_initializer=cfg.fc2_bias_initializer,
                use_bias=True,
            )
        )

        return ViTMLP(
            fc1=fc1,
            fc2=fc2,
            hidden_dim=cfg.vit_hidden_dim,
            mlp_hidden_dim=cfg.vit_mlp_hidden_dim,
            dropout=cfg.vit_dropout,
        )

    @classmethod
    def init(cls, key, mesh, rules, cfg):
        return super().init(key, mesh, rules, cfg)


def vit_mlp_forward(
    params: ViTMLP,
    x: Float[Array, "B S D"],
    key: jax.Array | None = None,
) -> Float[Array, "B S D"]:
    """
    Implementation of dropout based on flax NNX Dropout layer viz.
    (https://flax.readthedocs.io/en/latest/_modules/flax/nnx/nn/stochastic.html#Dropout)

    References:
    * https://github.com/AakashKumarNain/nanoGPTJAX/blob/main/nanogpt/model.py
    """
    with jax.named_scope("fc1"):
        x = linear_forward(params.fc1, x)  # (B, S, mlp_hidden_dim)

    with jax.named_scope("gelu"):
        x = jax.nn.gelu(x, approximate=True)  # (B, S, mlp_hidden_dim)

    with jax.named_scope("fc2"):
        x = linear_forward(params.fc2, x)  # (B, S, D)

    with jax.named_scope("dropout"):
        if key is not None:
            keep_prob = 1.0 - params.dropout
            dropout_mask = jax.random.bernoulli(key, keep_prob, x.shape)  # (B, S, D)
            x = x * dropout_mask / keep_prob  # (B, S, D)

    return x


#### Patch Embeddings ####


@jax_pytree_struct
class PatchEmbeddings(ParamInitializer):
    """

    References:
    * https://github.com/huggingface/nanoVLM/blob/main/models/vision_transformer.py
    """

    conv_weight: jax.Array | ParamSpec
    cls_token: jax.Array | ParamSpec | None
    position_embedding: jax.Array | ParamSpec
    img_size: int = dataclasses.field(metadata=dict(static=True))
    patch_size: int = dataclasses.field(metadata=dict(static=True))
    num_patches: int = dataclasses.field(metadata=dict(static=True))
    cls_flag: bool = dataclasses.field(metadata=dict(static=True))
    embd_dim: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        num_patches = (cfg.vit_img_size // cfg.vit_patch_size) ** 2

        conv_weight = ParamSpec(
            shape=(cfg.vit_patch_size, cfg.vit_patch_size, 3, cfg.vit_hidden_dim),
            dtype=cfg.dtype,
            logical_axes=cfg.conv_weight_logical_axes,
            initializer=cfg.conv_weight_initializer or _he_normal(),
        )

        if cfg.vit_cls_flag:
            cls_token = ParamSpec(
                shape=(1, 1, cfg.vit_hidden_dim),
                dtype=cfg.dtype,
                logical_axes=cfg.cls_token_logical_axes,
                initializer=cfg.cls_token_initializer or jax.nn.initializers.zeros,
            )
            position_embedding = ParamSpec(
                shape=(1, num_patches + 1, cfg.vit_hidden_dim),
                dtype=cfg.dtype,
                logical_axes=cfg.position_embedding_logical_axes,
                initializer=cfg.position_embedding_initializer or _embed_init,
            )
        else:
            cls_token = None
            position_embedding = ParamSpec(
                shape=(1, num_patches, cfg.vit_hidden_dim),
                dtype=cfg.dtype,
                logical_axes=cfg.position_embedding_logical_axes,
                initializer=cfg.position_embedding_initializer or _embed_init,
            )

        return PatchEmbeddings(
            conv_weight=conv_weight,
            cls_token=cls_token,
            position_embedding=position_embedding,
            img_size=cfg.vit_img_size,
            patch_size=cfg.vit_patch_size,
            num_patches=num_patches,
            cls_flag=cfg.vit_cls_flag,
            embd_dim=cfg.vit_hidden_dim,
        )

    @classmethod
    def init(cls, key, mesh, rules, cfg):
        return super().init(key, mesh, rules, cfg)


def conv2d_forward(
    params: Any,
    x: Float[Array, "B H W C_in"],
    stride: int,
    padding: str = "valid",
) -> Float[Array, "B H_out W_out C_out"]:
    return jax.lax.conv_general_dilated(
        x,
        params.weight,
        window_strides=(stride, stride),
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )


def patch_embeddings_forward(
    params: PatchEmbeddings,
    x: Float[Array, "B C H W"],
) -> Float[Array, "B S D"]:
    """
    References:
    * https://github.com/huggingface/nanoVLM/blob/main/models/vision_transformer.py
    """
    with jax.named_scope("conv"):
        # transpose to (B, H, W, C)
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = conv2d_forward(
            types.SimpleNamespace(weight=params.conv_weight),
            x,
            stride=params.patch_size,
            padding="valid",
        )  # (B, H', W', C')

    with jax.named_scope("flatten"):
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)  # (B, S, D)

    if params.cls_flag:
        with jax.named_scope("cls_token"):
            cls_token = jnp.broadcast_to(params.cls_token, (B, 1, C))  # (B, 1, D)
            x = jnp.concatenate([cls_token, x], axis=1)  # (B, S + 1, D)

    with jax.named_scope("position_embedding"):
        x = x + params.position_embedding  # (B, S, D) or (B, S + 1, D)

    return x


#### Multi-Head Attention ####


@jax_pytree_struct
class ViTMultiHeadAttention(ParamInitializer):
    """
    References:
    * https://github.com/huggingface/nanoVLM/blob/main/models/vision_transformer.py
    """

    qkv_proj: jax.Array | ParamSpec
    out_proj: jax.Array | ParamSpec
    num_heads: int = dataclasses.field(metadata=dict(static=True))
    embd_dim: int = dataclasses.field(metadata=dict(static=True))
    head_dim: int = dataclasses.field(metadata=dict(static=True))
    dropout: float = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        assert cfg.vit_hidden_dim % cfg.vit_num_heads == 0, (
            "embd_dim must be divisible by num_heads"
        )
        head_dim = cfg.vit_hidden_dim // cfg.vit_num_heads

        qkv_proj = ParamSpec(
            shape=(cfg.vit_hidden_dim, 3 * cfg.vit_hidden_dim),
            dtype=cfg.dtype,
            logical_axes=cfg.qkv_proj_logical_axes,
            initializer=cfg.qkv_proj_initializer
            or linear_init(cfg.vit_hidden_dim, 3 * cfg.vit_hidden_dim),
        )

        out_proj = ParamSpec(
            shape=(cfg.vit_hidden_dim, cfg.vit_hidden_dim),
            dtype=cfg.dtype,
            logical_axes=cfg.out_proj_logical_axes,
            initializer=cfg.out_proj_initializer
            or linear_init(cfg.vit_hidden_dim, cfg.vit_hidden_dim),
        )

        return ViTMultiHeadAttention(
            qkv_proj=qkv_proj,
            out_proj=out_proj,
            num_heads=cfg.vit_num_heads,
            embd_dim=cfg.vit_hidden_dim,
            head_dim=head_dim,
            dropout=cfg.vit_dropout,
        )

    @classmethod
    def init(cls, key, mesh, rules, cfg):
        return super().init(key, mesh, rules, cfg)


def vit_multi_head_attention_forward(
    params: ViTMultiHeadAttention,
    x: Float[Array, "B S D"],
    key: jax.Array | None = None,
) -> Float[Array, "B S D"]:
    """

    Implementation of dropout based on flax NNX Dropout layer viz.
    (https://flax.readthedocs.io/en/latest/_modules/flax/nnx/nn/stochastic.html#Dropout)

    References:
    * https://github.com/huggingface/nanoVLM/blob/main/models/vision_transformer.py
    """
    B, S, D = x.shape

    with jax.named_scope("qkv_proj"):
        qkv = jnp.dot(x, params.qkv_proj)  # (B, S, 3 * D)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each: (B, S, D)

    with jax.named_scope("reshape_heads"):
        q = q.reshape(B, S, params.num_heads, params.head_dim).transpose(
            0, 2, 1, 3
        )  # (B, num_heads, S, head_dim)
        k = k.reshape(B, S, params.num_heads, params.head_dim).transpose(
            0, 2, 1, 3
        )  # (B, num_heads, S, head_dim)
        v = v.reshape(B, S, params.num_heads, params.head_dim).transpose(
            0, 2, 1, 3
        )  # (B, num_heads, S, head_dim)

    with jax.named_scope("attention"):
        if key is not None:
            attn_key, resid_key = jax.random.split(key)
        else:
            attn_key = None
            resid_key = None

        scale = 1.0 / math.sqrt(k.shape[-1])
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale  # (B, num_heads, S, S)
        attn = jax.nn.softmax(attn, axis=-1)  # (B, num_heads, S, S)

        if attn_key is not None:
            keep_prob = 1.0 - params.dropout
            dropout_mask = jax.random.bernoulli(
                attn_key, keep_prob, attn.shape
            )  # (B, num_heads, S, S)
            attn = attn * dropout_mask / keep_prob  # (B, num_heads, S, S)

        y = jnp.einsum("bhqk,bhvd->bhqd", attn, v)  # (B, num_heads, S, head_dim)

    with jax.named_scope("reshape_combine"):
        y = y.transpose(0, 2, 1, 3)  # (B, S, num_heads, head_dim)
        y = y.reshape(B, S, D)  # (B, S, D)

    with jax.named_scope("out_proj"):
        y = jnp.dot(y, params.out_proj)  # (B, S, D)

    with jax.named_scope("resid_dropout"):
        if resid_key is not None:
            keep_prob = 1.0 - params.dropout
            dropout_mask = jax.random.bernoulli(
                resid_key, keep_prob, y.shape
            )  # (B, S, D)
            y = y * dropout_mask / keep_prob  # (B, S, D)

    return y


#### ViT Block ####


@jax_pytree_struct
class ViTBlock(ParamInitializer):
    """
    References:
    * https://github.com/huggingface/nanoVLM/blob/main/models/vision_transformer.py
    """

    ln1: LayerNorm
    attn: ViTMultiHeadAttention
    ln2: LayerNorm
    mlp: ViTMLP

    @classmethod
    def param_specs(cls, cfg):
        ln1 = LayerNorm.param_specs(
            types.SimpleNamespace(
                normalized_shape=cfg.vit_hidden_dim,
                eps=cfg.vit_ln_eps,
                dtype=cfg.dtype,
                scale_logical_axes=cfg.ln1_scale_logical_axes,
                bias_logical_axes=cfg.ln1_bias_logical_axes,
                scale_initializer=cfg.ln1_scale_initializer,
                bias_initializer=cfg.ln1_bias_initializer,
            )
        )

        attn = ViTMultiHeadAttention.param_specs(
            types.SimpleNamespace(
                vit_hidden_dim=cfg.vit_hidden_dim,
                vit_num_heads=cfg.vit_num_heads,
                vit_dropout=cfg.vit_dropout,
                dtype=cfg.dtype,
                qkv_proj_logical_axes=cfg.qkv_proj_logical_axes,
                out_proj_logical_axes=cfg.out_proj_logical_axes,
                qkv_proj_initializer=cfg.qkv_proj_initializer,
                out_proj_initializer=cfg.out_proj_initializer,
            )
        )

        ln2 = LayerNorm.param_specs(
            types.SimpleNamespace(
                normalized_shape=cfg.vit_hidden_dim,
                eps=cfg.vit_ln_eps,
                dtype=cfg.dtype,
                scale_logical_axes=cfg.ln2_scale_logical_axes,
                bias_logical_axes=cfg.ln2_bias_logical_axes,
                scale_initializer=cfg.ln2_scale_initializer,
                bias_initializer=cfg.ln2_bias_initializer,
            )
        )

        mlp = ViTMLP.param_specs(
            types.SimpleNamespace(
                vit_hidden_dim=cfg.vit_hidden_dim,
                vit_mlp_hidden_dim=cfg.vit_mlp_hidden_dim,
                vit_dropout=cfg.vit_dropout,
                dtype=cfg.dtype,
                fc1_weight_logical_axes=cfg.fc1_weight_logical_axes,
                fc1_bias_logical_axes=cfg.fc1_bias_logical_axes,
                fc2_weight_logical_axes=cfg.fc2_weight_logical_axes,
                fc2_bias_logical_axes=cfg.fc2_bias_logical_axes,
                fc1_weight_initializer=cfg.fc1_weight_initializer,
                fc1_bias_initializer=cfg.fc1_bias_initializer,
                fc2_weight_initializer=cfg.fc2_weight_initializer,
                fc2_bias_initializer=cfg.fc2_bias_initializer,
            )
        )

        return ViTBlock(
            ln1=ln1,
            attn=attn,
            ln2=ln2,
            mlp=mlp,
        )

    @classmethod
    def init(cls, key, mesh, rules, cfg):
        return super().init(key, mesh, rules, cfg)


def vit_block_forward(
    params: ViTBlock,
    x: Float[Array, "B S D"],
    key: jax.Array | None = None,
) -> Float[Array, "B S D"]:
    """
    References:
    * https://github.com/huggingface/nanoVLM/blob/main/models/vision_transformer.py
    """

    if key is not None:
        attn_key, mlp_key = jax.random.split(key)
    else:
        attn_key = None
        mlp_key = None

    with jax.named_scope("attn_residual"):
        x_norm = layer_norm_forward(params.ln1, x)  # (B, S, D)
        x = x + vit_multi_head_attention_forward(
            params.attn, x_norm, attn_key
        )  # (B, S, D)

    with jax.named_scope("mlp_residual"):
        x_norm = layer_norm_forward(params.ln2, x)  # (B, S, D)
        x = x + vit_mlp_forward(params.mlp, x_norm, mlp_key)  # (B, S, D)

    return x


#### ViT Model ####


@jax_pytree_struct
class ViT(ParamInitializer):
    """
    References:
    * https://github.com/huggingface/nanoVLM/blob/main/models/vision_transformer.py
    """

    patch_embedding: PatchEmbeddings
    blocks: tuple[ViTBlock, ...]
    layer_norm: LayerNorm
    cls_flag: bool = dataclasses.field(metadata=dict(static=True))
    dropout: float = dataclasses.field(metadata=dict(static=True))
    num_blocks: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        patch_embedding = PatchEmbeddings.param_specs(cfg)

        block_cfg = types.SimpleNamespace(
            vit_hidden_dim=cfg.vit_hidden_dim,
            vit_num_heads=cfg.vit_num_heads,
            vit_mlp_hidden_dim=cfg.vit_mlp_hidden_dim,
            vit_dropout=cfg.vit_dropout,
            vit_ln_eps=cfg.vit_ln_eps,
            dtype=cfg.dtype,
            ln1_scale_logical_axes=cfg.ln1_scale_logical_axes,
            ln1_bias_logical_axes=cfg.ln1_bias_logical_axes,
            ln2_scale_logical_axes=cfg.ln2_scale_logical_axes,
            ln2_bias_logical_axes=cfg.ln2_bias_logical_axes,
            qkv_proj_logical_axes=cfg.qkv_proj_logical_axes,
            out_proj_logical_axes=cfg.out_proj_logical_axes,
            fc1_weight_logical_axes=cfg.fc1_weight_logical_axes,
            fc1_bias_logical_axes=cfg.fc1_bias_logical_axes,
            fc2_weight_logical_axes=cfg.fc2_weight_logical_axes,
            fc2_bias_logical_axes=cfg.fc2_bias_logical_axes,
            ln1_scale_initializer=cfg.ln1_scale_initializer,
            ln1_bias_initializer=cfg.ln1_bias_initializer,
            ln2_scale_initializer=cfg.ln2_scale_initializer,
            ln2_bias_initializer=cfg.ln2_bias_initializer,
            qkv_proj_initializer=cfg.qkv_proj_initializer,
            out_proj_initializer=cfg.out_proj_initializer,
            fc1_weight_initializer=cfg.fc1_weight_initializer,
            fc1_bias_initializer=cfg.fc1_bias_initializer,
            fc2_weight_initializer=cfg.fc2_weight_initializer,
            fc2_bias_initializer=cfg.fc2_bias_initializer,
        )

        blocks = tuple(
            ViTBlock.param_specs(block_cfg) for _ in range(cfg.vit_num_blocks)
        )

        layer_norm = LayerNorm.param_specs(
            types.SimpleNamespace(
                normalized_shape=cfg.vit_hidden_dim,
                eps=cfg.vit_ln_eps,
                dtype=cfg.dtype,
                scale_logical_axes=cfg.final_ln_scale_logical_axes,
                bias_logical_axes=cfg.final_ln_bias_logical_axes,
                scale_initializer=cfg.final_ln_scale_initializer,
                bias_initializer=cfg.final_ln_bias_initializer,
            )
        )

        return ViT(
            patch_embedding=patch_embedding,
            blocks=blocks,
            layer_norm=layer_norm,
            cls_flag=cfg.vit_cls_flag,
            dropout=cfg.vit_dropout,
            num_blocks=cfg.vit_num_blocks,
        )

    @classmethod
    def init(cls, key, mesh, rules, cfg):
        return super().init(key, mesh, rules, cfg)

    @classmethod
    def from_pretrained(cls, model_id: str, *, revision: str = "main"):
        import json
        import os
        import types

        import jax.random as random
        import safetensors
        from huggingface_hub import hf_hub_download

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

        if "model_type" in hf_config and hf_config["model_type"] == "vit":
            cfg = types.SimpleNamespace(
                vit_model_type=model_id,
                vit_img_size=hf_config.get("vit_img_size", 224),
                vit_patch_size=hf_config.get("vit_patch_size", 16),
                vit_hidden_dim=hf_config.get("vit_hidden_dim", 768),
                vit_num_heads=hf_config.get("vit_num_heads", 12),
                vit_mlp_hidden_dim=hf_config.get("vit_mlp_hidden_dim", 3072),
                vit_num_blocks=hf_config.get("vit_num_blocks", 12),
                vit_cls_flag=hf_config.get("vit_cls_flag", False),
                vit_dropout=hf_config.get("vit_dropout", 0.0),
                vit_ln_eps=hf_config.get("vit_ln_eps", 1e-6),
                dtype=jnp.float32,
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
        else:
            vision_config = hf_config.get("vision_config", hf_config)
            text_config = hf_config.get("text_config", {})

            patch_size = vision_config.get("patch_size", 16)

            with safetensors.safe_open(
                safetensors_file, framework="pt", device="cpu"
            ) as f:
                hidden_dim = text_config.get("hidden_size", 768)
                num_heads = text_config.get("num_attention_heads", 12)
                mlp_hidden_dim = text_config.get("intermediate_size", 3072)

                num_layers = 0
                while (
                    f"vision_model.encoder.layers.{num_layers}.layer_norm1.weight"
                    in f.keys()
                ):
                    num_layers += 1
                if num_layers == 0:
                    num_layers = 12

            if "patch16" in model_id.lower():
                img_size = 224
            elif "patch32" in model_id.lower():
                img_size = 384
            else:
                img_size = 224

            cfg = types.SimpleNamespace(
                vit_model_type=model_id,
                vit_img_size=img_size,
                vit_patch_size=patch_size,
                vit_hidden_dim=hidden_dim,
                vit_num_heads=num_heads,
                vit_mlp_hidden_dim=mlp_hidden_dim,
                vit_num_blocks=num_layers,
                vit_cls_flag=False,
                vit_dropout=0.0,
                vit_ln_eps=1e-6,
                dtype=jnp.float32,
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

        hf_state_dict = {}
        with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                hf_state_dict[key] = f.get_tensor(key)

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
            elif isinstance(spec, tuple):
                return tuple(_init_pytree(item) for item in spec)
            return spec

        params = _init_pytree(param_specs)

        def _map_weights(hf_dict, params):
            conv_weight = jnp.array(
                hf_dict.pop("vision_model.embeddings.patch_embedding.weight")
            )
            hf_dict["patch_embedding.conv_weight"] = jnp.transpose(
                conv_weight, (2, 0, 1, 3)
            )

            hf_dict["patch_embedding.position_embedding"] = jnp.array(
                hf_dict.pop("vision_model.embeddings.position_embedding.weight")
            )

            if cfg.vit_cls_flag and "vision_model.embeddings.cls_token" in hf_dict:
                hf_dict["patch_embedding.cls_token"] = jnp.array(
                    hf_dict.pop("vision_model.embeddings.cls_token")
                )

            hf_dict["layer_norm.scale"] = jnp.array(
                hf_dict.pop("vision_model.post_layernorm.weight")
            )
            hf_dict["layer_norm.bias"] = jnp.array(
                hf_dict.pop("vision_model.post_layernorm.bias")
            )

            for i in range(cfg.vit_num_blocks):
                prefix = f"vision_model.encoder.layers.{i}"

                hf_dict[f"blocks.{i}.ln1.scale"] = jnp.array(
                    hf_dict.pop(f"{prefix}.layer_norm1.weight")
                )
                hf_dict[f"blocks.{i}.ln1.bias"] = jnp.array(
                    hf_dict.pop(f"{prefix}.layer_norm1.bias")
                )
                hf_dict[f"blocks.{i}.ln2.scale"] = jnp.array(
                    hf_dict.pop(f"{prefix}.layer_norm2.weight")
                )
                hf_dict[f"blocks.{i}.ln2.bias"] = jnp.array(
                    hf_dict.pop(f"{prefix}.layer_norm2.bias")
                )

                fc1_weight = jnp.array(hf_dict.pop(f"{prefix}.mlp.fc1.weight"))
                hf_dict[f"blocks.{i}.mlp.fc1.weight"] = fc1_weight.T
                hf_dict[f"blocks.{i}.mlp.fc1.bias"] = jnp.array(
                    hf_dict.pop(f"{prefix}.mlp.fc1.bias")
                )
                fc2_weight = jnp.array(hf_dict.pop(f"{prefix}.mlp.fc2.weight"))
                hf_dict[f"blocks.{i}.mlp.fc2.weight"] = fc2_weight.T
                hf_dict[f"blocks.{i}.mlp.fc2.bias"] = jnp.array(
                    hf_dict.pop(f"{prefix}.mlp.fc2.bias")
                )

                q_w = jnp.array(hf_dict.pop(f"{prefix}.self_attn.q_proj.weight"))
                k_w = jnp.array(hf_dict.pop(f"{prefix}.self_attn.k_proj.weight"))
                v_w = jnp.array(hf_dict.pop(f"{prefix}.self_attn.v_proj.weight"))
                hf_dict[f"blocks.{i}.attn.qkv_proj.weight"] = jnp.concatenate(
                    [q_w, k_w, v_w], axis=0
                ).T

                if (
                    f"{prefix}.self_attn.q_proj.bias" in hf_dict
                    and f"{prefix}.self_attn.k_proj.bias" in hf_dict
                    and f"{prefix}.self_attn.v_proj.bias" in hf_dict
                ):
                    q_b = jnp.array(hf_dict.pop(f"{prefix}.self_attn.q_proj.bias"))
                    k_b = jnp.array(hf_dict.pop(f"{prefix}.self_attn.k_proj.bias"))
                    v_b = jnp.array(hf_dict.pop(f"{prefix}.self_attn.v_proj.bias"))
                    hf_dict[f"blocks.{i}.attn.qkv_proj.bias"] = jnp.concatenate(
                        [q_b, k_b, v_b], axis=0
                    )

                out_proj_weight = jnp.array(
                    hf_dict.pop(f"{prefix}.self_attn.out_proj.weight")
                )
                hf_dict[f"blocks.{i}.attn.out_proj.weight"] = out_proj_weight.T
                if f"{prefix}.self_attn.out_proj.bias" in hf_dict:
                    hf_dict[f"blocks.{i}.attn.out_proj.bias"] = jnp.array(
                        hf_dict.pop(f"{prefix}.self_attn.out_proj.bias")
                    )

            def _set_nested_attr(obj, path, value):
                parts = path.split(".")
                for part in parts[:-1]:
                    if part == "blocks":
                        idx = int(parts[1])
                        obj = obj.blocks[idx]
                    else:
                        obj = getattr(obj, part)
                setattr(obj, parts[-1], value)

            for jax_key, value in hf_dict.items():
                if not jax_key.startswith("vision_model.") and not jax_key.startswith(
                    "blocks."
                ):
                    continue
                try:
                    _set_nested_attr(params, jax_key, value)
                except AttributeError as e:
                    logger.warning(
                        "Failed to map weight '%s' from checkpoint: %s", jax_key, e
                    )

            return params

        return _map_weights(hf_state_dict, params)

    def save_pretrained(self, save_dir: str):
        import json
        import os

        import safetensors.numpy

        os.makedirs(save_dir, exist_ok=True)

        state_dict = {}

        state_dict["vision_model.embeddings.patch_embedding.weight"] = np.array(
            self.patch_embedding.conv_weight
        )
        state_dict["vision_model.embeddings.position_embedding.weight"] = np.array(
            self.patch_embedding.position_embedding
        )

        if self.patch_embedding.cls_token is not None:
            state_dict["vision_model.embeddings.cls_token"] = np.array(
                self.patch_embedding.cls_token
            )

        state_dict["vision_model.post_layernorm.weight"] = np.array(
            self.layer_norm.scale
        )
        state_dict["vision_model.post_layernorm.bias"] = np.array(self.layer_norm.bias)

        for i, block in enumerate(self.blocks):
            prefix = f"vision_model.encoder.layers.{i}"

            state_dict[f"{prefix}.layer_norm1.weight"] = np.array(block.ln1.scale)
            state_dict[f"{prefix}.layer_norm1.bias"] = np.array(block.ln1.bias)
            state_dict[f"{prefix}.layer_norm2.weight"] = np.array(block.ln2.scale)
            state_dict[f"{prefix}.layer_norm2.bias"] = np.array(block.ln2.bias)

            if hasattr(block.mlp.fc1, "weight"):
                fc1_weight = np.array(block.mlp.fc1.weight).T
                fc1_bias = np.array(block.mlp.fc1.bias)
                fc2_weight = np.array(block.mlp.fc2.weight).T
                fc2_bias = np.array(block.mlp.fc2.bias)
            else:
                fc1_weight = np.array(block.mlp.fc1).T
                fc1_bias = None
                fc2_weight = np.array(block.mlp.fc2).T
                fc2_bias = None

            state_dict[f"{prefix}.mlp.fc1.weight"] = fc1_weight
            if fc1_bias is not None:
                state_dict[f"{prefix}.mlp.fc1.bias"] = fc1_bias
            state_dict[f"{prefix}.mlp.fc2.weight"] = fc2_weight
            if fc2_bias is not None:
                state_dict[f"{prefix}.mlp.fc2.bias"] = fc2_bias

            if hasattr(block.attn.qkv_proj, "weight"):
                qkv_proj_array = block.attn.qkv_proj.weight
                qkv_bias_array = block.attn.qkv_proj.bias
            else:
                qkv_proj_array = block.attn.qkv_proj
                qkv_bias_array = None

            qkv_weight = np.array(qkv_proj_array).T
            hidden_dim = qkv_weight.shape[0] // 3
            q_w = qkv_weight[:hidden_dim, :]
            k_w = qkv_weight[hidden_dim : 2 * hidden_dim, :]
            v_w = qkv_weight[2 * hidden_dim :, :]

            state_dict[f"{prefix}.self_attn.q_proj.weight"] = q_w
            state_dict[f"{prefix}.self_attn.k_proj.weight"] = k_w
            state_dict[f"{prefix}.self_attn.v_proj.weight"] = v_w

            if qkv_bias_array is not None:
                qkv_bias = np.array(qkv_bias_array)
                q_b = qkv_bias[:hidden_dim]
                k_b = qkv_bias[hidden_dim : 2 * hidden_dim]
                v_b = qkv_bias[2 * hidden_dim :]

                state_dict[f"{prefix}.self_attn.q_proj.bias"] = q_b
                state_dict[f"{prefix}.self_attn.k_proj.bias"] = k_b
                state_dict[f"{prefix}.self_attn.v_proj.bias"] = v_b

            if hasattr(block.attn.out_proj, "weight"):
                out_proj_weight = np.array(block.attn.out_proj.weight).T
                out_proj_bias = np.array(block.attn.out_proj.bias)
            else:
                out_proj_weight = np.array(block.attn.out_proj).T
                out_proj_bias = None

            state_dict[f"{prefix}.self_attn.out_proj.weight"] = out_proj_weight
            if out_proj_bias is not None:
                state_dict[f"{prefix}.self_attn.out_proj.bias"] = out_proj_bias

        safetensors.numpy.save_file(
            state_dict, os.path.join(save_dir, "model.safetensors")
        )

        config = {
            "model_type": "vit",
            "vit_img_size": self.patch_embedding.img_size,
            "vit_patch_size": self.patch_embedding.patch_size,
            "vit_hidden_dim": self.patch_embedding.embd_dim,
            "vit_num_heads": self.blocks[0].attn.num_heads,
            "vit_num_blocks": self.num_blocks,
            "vit_cls_flag": self.cls_flag,
            "vit_ln_eps": self.blocks[0].ln1.eps,
        }

        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def push_to_hub(
        self,
        model_id: str,
        *,
        private: bool = False,
        commit_message: str = "Upload ViT model",
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


def vit_forward(
    params: ViT,
    x: Float[Array, "B C H W"],
    key: jax.Array | None = None,
) -> Float[Array, "B D"] | Float[Array, "B S D"]:
    """
    References:
    * https://github.com/huggingface/nanoVLM/blob/main/models/vision_transformer.py
    """

    if key is not None:
        keys = jax.random.split(key, params.num_blocks + 1)
        embed_key = keys[0]
        block_keys = keys[1:]
    else:
        embed_key = None
        block_keys = [None] * params.num_blocks

    with jax.named_scope("patch_embedding"):
        x = patch_embeddings_forward(params.patch_embedding, x)  # (B, S, D)

    with jax.named_scope("embedding_dropout"):
        if embed_key is not None:
            keep_prob = 1.0 - params.dropout
            dropout_mask = jax.random.bernoulli(
                embed_key, keep_prob, x.shape
            )  # (B, S, D)
            x = x * dropout_mask / keep_prob  # (B, S, D)

    with jax.named_scope("blocks"):
        for block, block_key in zip(params.blocks, block_keys):
            x = vit_block_forward(block, x, block_key)  # (B, S, D)

    with jax.named_scope("final_layer_norm"):
        if params.cls_flag:
            x = x[:, 0]  # (B, D)
        x = layer_norm_forward(params.layer_norm, x)  # (B, D) or (B, S, D)

    return x
