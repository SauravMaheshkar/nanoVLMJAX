import dataclasses
from functools import lru_cache
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.utils import ParamInitializer, ParamSpec, jax_pytree_struct

_he_normal = lru_cache(jax.nn.initializers.he_normal)


def linear_init(fan_in, fan_out) -> Callable:
    import math

    std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
    return jax.nn.initializers.normal(stddev=std)


#### Modality Projector ####


@jax_pytree_struct
class ModalityProjector(ParamInitializer):
    """
    Projects vision embeddings to language model embedding space.

    Uses pixel shuffle to reshape vision features and then projects to the
    language model hidden dimension.

    References:
        * https://github.com/huggingface/nanoVLM/blob/main/models/modality_projector.py
    """

    proj: jax.Array | ParamSpec
    input_dim: int = dataclasses.field(metadata=dict(static=True))
    output_dim: int = dataclasses.field(metadata=dict(static=True))
    scale_factor: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        input_dim = cfg.vit_hidden_dim * (cfg.mp_pixel_shuffle_factor**2)
        output_dim = cfg.lm_hidden_dim

        proj = ParamSpec(
            shape=(input_dim, output_dim),
            dtype=cfg.dtype,
            logical_axes=cfg.mp_proj_logical_axes,
            initializer=cfg.mp_proj_initializer or linear_init(input_dim, output_dim),
        )

        return ModalityProjector(
            proj=proj,
            input_dim=input_dim,
            output_dim=output_dim,
            scale_factor=cfg.mp_pixel_shuffle_factor,
        )

    @classmethod
    def init(cls, key, mesh, rules, cfg):
        return super().init(key, mesh, rules, cfg)


def pixel_shuffle(
    x: Float[Array, "B S D"], scale_factor: int
) -> Float[Array, "B new_seq new_dim"]:
    """
    Performs pixel shuffle operation to downsample spatial dimensions.

    Args:
        x: Input tensor of shape (B, S, D)
        scale_factor: The factor to reduce spatial dimensions by

    Returns:
        Reshaped tensor with reduced spatial dimensions and increased channels
    """
    B, seq, embed_dim = x.shape
    seq_root = int(seq**0.5)
    assert seq_root * seq_root == seq, (
        "Sequence length must be a perfect square for pixel shuffle"
    )
    assert seq_root % scale_factor == 0, (
        "Sequence root must be divisible by scale factor"
    )

    height = width = seq_root
    h_out = height // scale_factor
    w_out = width // scale_factor

    x = x.reshape(B, height, width, embed_dim)
    x = x.reshape(B, h_out, scale_factor, w_out, scale_factor, embed_dim)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    x = x.reshape(B, h_out * w_out, embed_dim * scale_factor * scale_factor)

    return x


def modality_projector_forward(
    params: ModalityProjector,
    x: Float[Array, "B S D"],
) -> Float[Array, "B S' O"]:
    """
    Forward pass through the modality projector.

    Args:
        params: ModalityProjector parameters
        x: Input tensor of shape (B, S, D) from vision encoder

    Returns:
        Projected tensor of shape (B, S', V) where S' is reduced spatial dimension
    """
    with jax.named_scope("pixel_shuffle"):
        x = pixel_shuffle(x, params.scale_factor)

    with jax.named_scope("proj"):
        x = jnp.dot(x, params.proj)

    return x
