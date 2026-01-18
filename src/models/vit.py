import dataclasses
import math
import types
from functools import lru_cache
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.utils import ParamInitializer, ParamSpec, jax_pytree_struct

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
