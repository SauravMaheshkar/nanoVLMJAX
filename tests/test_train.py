import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from configs.default import TrainConfig
from src.input_pipeline import collate_vlm_batch
from src.models.vision_language_model import (
    VisionLanguageModel,
    VLMConfig,
)
from src.opt_factory import get_optimizer
from tests.conftest import (
    assert_on_mesh_devices,
    maybe_mesh,
    shard_batch,
)
from train import eval_step, train_step


@pytest.fixture
def tiny_vlm_cfg():
    return VLMConfig(
        vit_hidden_dim=64,
        vit_inter_dim=128,
        vit_patch_size=16,
        vit_img_size=32,
        vit_n_heads=4,
        vit_dropout=0.0,
        vit_n_blocks=2,
        vit_ln_eps=1e-6,
        vit_cls_flag=False,
        lm_hidden_dim=64,
        lm_inter_dim=128,
        lm_rms_eps=1e-5,
        lm_re_base=100000,
        lm_max_position_embeddings=256,
        lm_vocab_size=256,
        lm_n_heads=4,
        lm_n_kv_heads=2,
        lm_dropout=0.0,
        lm_n_layers=2,
        lm_attn_scaling=1.0,
        lm_use_tokens=False,
        lm_tie_weights=True,
        lm_attention_bias=False,
        mp_pixel_shuffle_factor=2,
        mp_image_token_length=4,
        image_token_id=10,
        dtype=jnp.float32,
    )


class ShardingRule:
    batch = "fsdp"
    seq = None
    hidden = "tp"
    tp = "tp"
    kernel_h = None
    kernel_w = None
    in_channels = None
    out_channels = "tp"


@pytest.fixture
def tiny_params(mesh, tiny_vlm_cfg):
    key = jax.random.PRNGKey(42)
    return VisionLanguageModel.init(key, mesh, ShardingRule, tiny_vlm_cfg)


def test_get_optimizer(tiny_params):
    optimizer = get_optimizer(
        tiny_params,
        lr_mp=1e-3,
        lr_vision=1e-4,
        lr_language=1e-5,
        max_steps=100,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
    )
    opt_state = optimizer.init(tiny_params)
    assert opt_state is not None


def test_train_step(tiny_params, tiny_vlm_cfg):
    optimizer = get_optimizer(
        tiny_params,
        lr_mp=1e-3,
        lr_vision=1e-4,
        lr_language=1e-5,
        max_steps=100,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
    )
    opt_state = optimizer.init(tiny_params)

    batch = {
        "input_ids": jnp.ones((2, 16), dtype=jnp.int32),
        "attention_mask": jnp.ones((2, 16), dtype=jnp.int32),
        "labels": jnp.ones((2, 16), dtype=jnp.int32),
        "images": jnp.ones(
            (2, 3, tiny_vlm_cfg.vit_img_size, tiny_vlm_cfg.vit_img_size),
            dtype=jnp.float32,
        ),
    }
    key = jax.random.PRNGKey(0)

    new_params, new_opt_state, loss = train_step(
        tiny_params,
        opt_state,
        batch,
        key,
        optimizer,
        use_mixed_precision=False,
    )
    assert loss is not None
    assert not jnp.isnan(loss)
    assert new_params is not tiny_params


def test_eval_step(tiny_params, tiny_vlm_cfg):
    batch = {
        "input_ids": jnp.ones((2, 16), dtype=jnp.int32),
        "attention_mask": jnp.ones((2, 16), dtype=jnp.int32),
        "labels": jnp.ones((2, 16), dtype=jnp.int32),
        "images": jnp.ones(
            (2, 3, tiny_vlm_cfg.vit_img_size, tiny_vlm_cfg.vit_img_size),
            dtype=jnp.float32,
        ),
    }
    loss = eval_step(tiny_params, batch, use_mixed_precision=False)
    assert loss is not None
    assert not jnp.isnan(loss)


def test_collate_vlm_batch():
    pad_token_id = 0
    samples = [
        {
            "input_ids": np.array([1, 2, 3], dtype=np.int32),
            "attention_mask": np.array([1, 1, 1], dtype=np.int32),
            "labels": np.array([2, 3, -100], dtype=np.int32),
            "images": [{"pixel_values": np.ones((1, 3, 224, 224), dtype=np.float32)}],
        },
        {
            "input_ids": np.array([4, 5], dtype=np.int32),
            "attention_mask": np.array([1, 1], dtype=np.int32),
            "labels": np.array([5, -100], dtype=np.int32),
            "images": [],
        },
    ]

    batch = collate_vlm_batch(samples, pad_token_id, max_length=8)
    assert batch is not None
    assert batch["input_ids"].shape == (2, 8)
    assert batch["attention_mask"].shape == (2, 8)
    assert batch["labels"].shape == (2, 8)
    assert batch["images"].shape == (2, 3, 224, 224)

    # Check left-padding
    assert batch["input_ids"][0, 0] == pad_token_id
    assert batch["input_ids"][0, -3] == 1


def test_collate_vlm_batch_max_length_filter():
    pad_token_id = 0
    samples = [
        {
            "input_ids": np.array([1] * 10, dtype=np.int32),
            "attention_mask": np.array([1] * 10, dtype=np.int32),
            "labels": np.array([1] * 10, dtype=np.int32),
            "images": [],
        },
        {
            "input_ids": np.array([2] * 5, dtype=np.int32),
            "attention_mask": np.array([1] * 5, dtype=np.int32),
            "labels": np.array([2] * 5, dtype=np.int32),
            "images": [],
        },
    ]
    batch = collate_vlm_batch(samples, pad_token_id, max_length=6)
    assert batch is not None
    assert batch["input_ids"].shape == (1, 6)


def test_save_and_resume_checkpoint(tiny_params, tiny_vlm_cfg):
    with tempfile.TemporaryDirectory() as tmpdir:
        tiny_params.save_pretrained(tmpdir)
        loaded = VisionLanguageModel.from_pretrained(tmpdir)
        assert isinstance(loaded, VisionLanguageModel)
        assert loaded.decoder.vocab_size == tiny_params.decoder.vocab_size


def test_train_config_replace():
    cfg = TrainConfig()
    cfg2 = cfg.replace(lr_mp=1e-2)
    assert cfg2.lr_mp == 1e-2
    assert cfg.lr_mp != 1e-2


@pytest.mark.parametrize(
    "mesh_shape", [None, (1, 8), (2, 4)], ids=["no_mesh", "1x8", "2x4"]
)
def test_train_step_multi_device(mesh_shape, jax_devices, tiny_vlm_cfg):
    """train_step on different mesh configurations including single-device."""
    key = jax.random.PRNGKey(42)
    if mesh_shape is None:
        # init requires a mesh, so use a single-device mesh for init, then
        # run the actual step without a mesh context (non-distributed path).
        init_mesh = jax.sharding.Mesh(
            np.array([jax_devices[0]]).reshape(1, 1),
            axis_names=("fsdp", "tp"),
        )
        params = VisionLanguageModel.init(key, init_mesh, ShardingRule, tiny_vlm_cfg)
        mesh = None
    else:
        fsdp, tp = mesh_shape
        mesh = jax.sharding.Mesh(
            np.array(jax_devices).reshape(fsdp, tp),
            axis_names=("fsdp", "tp"),
        )
        params = VisionLanguageModel.init(key, mesh, ShardingRule, tiny_vlm_cfg)

    optimizer = get_optimizer(
        params,
        lr_mp=1e-3,
        lr_vision=1e-4,
        lr_language=1e-5,
        max_steps=100,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
    )
    opt_state = optimizer.init(params)

    batch = {
        "input_ids": jnp.ones((8, 16), dtype=jnp.int32),
        "attention_mask": jnp.ones((8, 16), dtype=jnp.int32),
        "labels": jnp.ones((8, 16), dtype=jnp.int32),
        "images": jnp.ones(
            (8, 3, tiny_vlm_cfg.vit_img_size, tiny_vlm_cfg.vit_img_size),
            dtype=jnp.float32,
        ),
    }
    batch = shard_batch(batch, mesh)
    key = jax.random.PRNGKey(0)

    if mesh is not None:
        batch_devices = batch["input_ids"].sharding.device_set
        assert len(batch_devices) > 1, "Batch should be sharded across multiple devices"

    with maybe_mesh(mesh):
        new_params, new_opt_state, loss = train_step(
            params,
            opt_state,
            batch,
            key,
            optimizer,
            use_mixed_precision=False,
        )

    assert loss is not None
    assert not jnp.isnan(loss)

    for leaf in jax.tree.leaves(new_params):
        assert_on_mesh_devices(leaf, mesh)


@pytest.mark.parametrize(
    "mesh_shape", [None, (1, 8), (2, 4)], ids=["no_mesh", "1x8", "2x4"]
)
def test_eval_step_multi_device(mesh_shape, jax_devices, tiny_vlm_cfg):
    """eval_step on different mesh configurations including single-device."""
    key = jax.random.PRNGKey(42)
    if mesh_shape is None:
        init_mesh = jax.sharding.Mesh(
            np.array([jax_devices[0]]).reshape(1, 1),
            axis_names=("fsdp", "tp"),
        )
        params = VisionLanguageModel.init(key, init_mesh, ShardingRule, tiny_vlm_cfg)
        mesh = None
    else:
        fsdp, tp = mesh_shape
        mesh = jax.sharding.Mesh(
            np.array(jax_devices).reshape(fsdp, tp),
            axis_names=("fsdp", "tp"),
        )
        params = VisionLanguageModel.init(key, mesh, ShardingRule, tiny_vlm_cfg)

    batch = {
        "input_ids": jnp.ones((8, 16), dtype=jnp.int32),
        "attention_mask": jnp.ones((8, 16), dtype=jnp.int32),
        "labels": jnp.ones((8, 16), dtype=jnp.int32),
        "images": jnp.ones(
            (8, 3, tiny_vlm_cfg.vit_img_size, tiny_vlm_cfg.vit_img_size),
            dtype=jnp.float32,
        ),
    }
    batch = shard_batch(batch, mesh)

    if mesh is not None:
        batch_devices = batch["input_ids"].sharding.device_set
        assert len(batch_devices) > 1, "Batch should be sharded across multiple devices"

    with maybe_mesh(mesh):
        loss = eval_step(params, batch, use_mixed_precision=False)

    assert loss is not None
    assert not jnp.isnan(loss)
