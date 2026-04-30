import dataclasses

import jax
import jax.numpy as jnp
import jax.random as random
import pytest

from src.models.modality_projector import (
    ModalityProjector,
    modality_projector_forward,
    pixel_shuffle,
)


@dataclasses.dataclass
class ModalityProjectorConfig:
    vit_hidden_dim: int = 768
    lm_hidden_dim: int = 576
    mp_pixel_shuffle_factor: int = 2
    dtype: jnp.dtype = jnp.float32
    mp_proj_logical_axes: tuple = (None, "tp")
    mp_proj_initializer = None


class ShardingRule:
    batch = "fsdp"
    seq = None
    hidden = "tp"
    tp = "tp"


@pytest.mark.parametrize(
    "batch_size,seq_len,hidden_dim,scale_factor",
    [
        (1, 16, 64, 2),
    ],
    ids=["with-jit"],
)
def test_modality_projector_forward(
    mesh, batch_size, seq_len, hidden_dim, scale_factor
):
    cfg = ModalityProjectorConfig(
        vit_hidden_dim=hidden_dim,
        lm_hidden_dim=64,
        mp_pixel_shuffle_factor=scale_factor,
    )

    key = random.PRNGKey(42)
    params = ModalityProjector.init(key, mesh, ShardingRule, cfg)
    x = random.normal(key, (batch_size, seq_len, hidden_dim))

    forward_fn = jax.jit(modality_projector_forward)
    output = forward_fn(params, x)

    seq_root = int(seq_len**0.5)
    expected_seq_len = (seq_root // scale_factor) ** 2

    assert output.shape == (batch_size, expected_seq_len, cfg.lm_hidden_dim)
    assert output.dtype == cfg.dtype


def test_pixel_shuffle():
    B, S, D = 1, 16, 64
    scale_factor = 2
    x = random.normal(random.PRNGKey(0), (B, S, D))

    output = pixel_shuffle(x, scale_factor)

    seq_root = int(S**0.5)
    expected_seq = (seq_root // scale_factor) ** 2
    expected_dim = D * (scale_factor**2)

    assert output.shape == (B, expected_seq, expected_dim)
