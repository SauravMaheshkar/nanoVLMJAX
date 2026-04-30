import dataclasses

import grain
import jax
import jax.numpy as jnp
import jax.random as random
import pytest

from src.models.vit import (
    LayerNorm,
    PatchEmbeddings,
    ViT,
    ViTBlock,
    ViTMLP,
    ViTMultiHeadAttention,
    layer_norm_forward,
    patch_embeddings_forward,
    vit_block_forward,
    vit_forward,
    vit_mlp_forward,
    vit_multi_head_attention_forward,
)


@dataclasses.dataclass
class ExampleConfig:
    vit_img_size: int = 224
    vit_patch_size: int = 16
    vit_hidden_dim: int = 768
    vit_cls_flag: bool = True
    dtype: jnp.dtype = jnp.float32
    conv_weight_logical_axes: tuple = (
        "kernel_h",
        "kernel_w",
        "in_channels",
        "out_channels",
    )
    cls_token_logical_axes: tuple = (None, "seq", "hidden")
    position_embedding_logical_axes: tuple = (None, "seq", "hidden")
    conv_weight_initializer = None
    cls_token_initializer = None
    position_embedding_initializer = None


class ShardingRule:
    kernel_h = None
    kernel_w = None
    in_channels = None
    out_channels = "tp"
    batch = "fsdp"
    seq = None
    hidden = "tp"


@pytest.mark.parametrize(
    "img_size,patch_size,hidden_dim,cls_flag,batch_size",
    [
        (64, 8, 192, True, 1),
    ],
    ids=["with-cls-flag"],
)
def test_patch_embeddings_forward(
    mesh, img_size, patch_size, hidden_dim, cls_flag, batch_size
):
    cfg = ExampleConfig(
        vit_img_size=img_size,
        vit_patch_size=patch_size,
        vit_hidden_dim=hidden_dim,
        vit_cls_flag=cls_flag,
    )

    key = random.PRNGKey(42)
    params = PatchEmbeddings.init(key, mesh, ShardingRule, cfg)
    x = random.normal(key, (batch_size, 3, img_size, img_size))

    forward_fn = jax.jit(patch_embeddings_forward)
    output1 = forward_fn(params, x)
    output2 = forward_fn(params, x)
    assert jnp.allclose(output1, output2)

    num_patches = (img_size // patch_size) ** 2
    expected_seq_len = num_patches + 1 if cls_flag else num_patches
    assert output1.shape == (batch_size, expected_seq_len, hidden_dim)
    assert output1.dtype == cfg.dtype


@dataclasses.dataclass
class AttentionConfig:
    vit_hidden_dim: int = 768
    vit_num_heads: int = 12
    vit_dropout: float = 0.1
    dtype: jnp.dtype = jnp.float32
    qkv_proj_logical_axes: tuple = ("hidden", None)
    out_proj_logical_axes: tuple = ("hidden", None)
    qkv_proj_initializer = None
    out_proj_initializer = None


@pytest.mark.parametrize(
    "hidden_dim,num_heads,batch_size,seq_len",
    [
        (192, 3, 1, 17),
    ],
    ids=["with-jit"],
)
def test_attention_forward(mesh, hidden_dim, num_heads, batch_size, seq_len):
    cfg = AttentionConfig(vit_hidden_dim=hidden_dim, vit_num_heads=num_heads)

    key = random.PRNGKey(42)
    params = ViTMultiHeadAttention.init(key, mesh, ShardingRule, cfg)
    x = random.normal(key, (batch_size, seq_len, hidden_dim))

    key = random.PRNGKey(43)
    forward_fn = jax.jit(vit_multi_head_attention_forward)
    output1 = forward_fn(params, x, key)
    output2 = forward_fn(params, x, key)
    assert jnp.allclose(output1, output2)

    assert output1.shape == (batch_size, seq_len, hidden_dim)
    assert output1.dtype == cfg.dtype


@dataclasses.dataclass
class MLPConfig:
    vit_hidden_dim: int = 768
    vit_mlp_hidden_dim: int = 3072
    vit_dropout: float = 0.1
    dtype: jnp.dtype = jnp.float32
    fc1_weight_logical_axes: tuple = ("hidden", None)
    fc1_bias_logical_axes: tuple = (None,)
    fc2_weight_logical_axes: tuple = (None, "hidden")
    fc2_bias_logical_axes: tuple = ("hidden",)
    fc1_weight_initializer = None
    fc1_bias_initializer = None
    fc2_weight_initializer = None
    fc2_bias_initializer = None


@pytest.mark.parametrize(
    "hidden_dim,mlp_hidden_dim,batch_size,seq_len",
    [
        (192, 768, 1, 17),
    ],
    ids=["with-jit"],
)
def test_mlp_forward(mesh, hidden_dim, mlp_hidden_dim, batch_size, seq_len):
    cfg = MLPConfig(vit_hidden_dim=hidden_dim, vit_mlp_hidden_dim=mlp_hidden_dim)

    key = random.PRNGKey(42)
    params = ViTMLP.init(key, mesh, ShardingRule, cfg)
    x = random.normal(key, (batch_size, seq_len, hidden_dim))

    key = random.PRNGKey(43)
    forward_fn = jax.jit(vit_mlp_forward)
    output1 = forward_fn(params, x, key)
    output2 = forward_fn(params, x, key)
    assert jnp.allclose(output1, output2)

    assert output1.shape == (batch_size, seq_len, hidden_dim)
    assert output1.dtype == cfg.dtype


@dataclasses.dataclass
class LayerNormConfig:
    normalized_shape: int = 768
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    scale_logical_axes: tuple = ("hidden",)
    bias_logical_axes: tuple = ("hidden",)
    scale_initializer = None
    bias_initializer = None


@pytest.mark.parametrize(
    "normalized_shape",
    [
        192,
    ],
    ids=["192"],
)
def test_layer_norm_init(mesh, normalized_shape):
    cfg = LayerNormConfig(normalized_shape=normalized_shape)

    key = random.PRNGKey(42)
    params = LayerNorm.init(key, mesh, ShardingRule, cfg)

    assert isinstance(params, LayerNorm)
    assert params.scale.shape == (normalized_shape,)
    assert params.bias.shape == (normalized_shape,)
    assert params.normalized_shape == normalized_shape
    assert params.eps == cfg.eps


@pytest.mark.parametrize(
    "normalized_shape,batch_size,seq_len",
    [
        (64, 1, 17),
    ],
    ids=["with-jit"],
)
def test_layer_norm_forward(mesh, normalized_shape, batch_size, seq_len):
    cfg = LayerNormConfig(normalized_shape=normalized_shape)

    key = random.PRNGKey(42)
    params = LayerNorm.init(key, mesh, ShardingRule, cfg)
    x = random.normal(key, (batch_size, seq_len, normalized_shape))

    forward_fn = jax.jit(layer_norm_forward)
    output1 = forward_fn(params, x)
    output2 = forward_fn(params, x)
    assert jnp.allclose(output1, output2)

    assert output1.shape == (batch_size, seq_len, normalized_shape)
    assert output1.dtype == cfg.dtype


@dataclasses.dataclass
class ViTBlockConfig:
    vit_hidden_dim: int = 768
    vit_num_heads: int = 12
    vit_mlp_hidden_dim: int = 3072
    vit_dropout: float = 0.1
    vit_ln_eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    ln1_scale_logical_axes: tuple = ("hidden",)
    ln1_bias_logical_axes: tuple = ("hidden",)
    ln2_scale_logical_axes: tuple = ("hidden",)
    ln2_bias_logical_axes: tuple = ("hidden",)
    qkv_proj_logical_axes: tuple = ("hidden", None)
    out_proj_logical_axes: tuple = ("hidden", None)
    fc1_weight_logical_axes: tuple = ("hidden", None)
    fc1_bias_logical_axes: tuple = (None,)
    fc2_weight_logical_axes: tuple = (None, "hidden")
    fc2_bias_logical_axes: tuple = ("hidden",)
    ln1_scale_initializer = None
    ln1_bias_initializer = None
    ln2_scale_initializer = None
    ln2_bias_initializer = None
    qkv_proj_initializer = None
    out_proj_initializer = None
    fc1_weight_initializer = None
    fc1_bias_initializer = None
    fc2_weight_initializer = None
    fc2_bias_initializer = None


@pytest.mark.parametrize(
    "hidden_dim,num_heads,mlp_hidden_dim,batch_size,seq_len",
    [
        (192, 3, 768, 1, 17),
    ],
    ids=["with-jit"],
)
def test_vit_block_forward(
    mesh,
    hidden_dim,
    num_heads,
    mlp_hidden_dim,
    batch_size,
    seq_len,
):
    cfg = ViTBlockConfig(
        vit_hidden_dim=hidden_dim,
        vit_num_heads=num_heads,
        vit_mlp_hidden_dim=mlp_hidden_dim,
    )

    key = random.PRNGKey(42)
    params = ViTBlock.init(key, mesh, ShardingRule, cfg)
    x = random.normal(key, (batch_size, seq_len, hidden_dim))

    key = random.PRNGKey(43)
    forward_fn = jax.jit(vit_block_forward)
    output1 = forward_fn(params, x, key)
    output2 = forward_fn(params, x, key)
    assert jnp.allclose(output1, output2)

    assert output1.shape == (batch_size, seq_len, hidden_dim)
    assert output1.dtype == cfg.dtype


@dataclasses.dataclass
class ViTConfig:
    vit_img_size: int = 64
    vit_patch_size: int = 8
    vit_hidden_dim: int = 192
    vit_num_heads: int = 3
    vit_mlp_hidden_dim: int = 768
    vit_num_blocks: int = 2
    vit_cls_flag: bool = True
    vit_dropout: float = 0.1
    vit_ln_eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    conv_weight_logical_axes: tuple = (
        "kernel_h",
        "kernel_w",
        "in_channels",
        "out_channels",
    )
    cls_token_logical_axes: tuple = (None, "seq", "hidden")
    position_embedding_logical_axes: tuple = (None, "seq", "hidden")
    ln1_scale_logical_axes: tuple = ("hidden",)
    ln1_bias_logical_axes: tuple = ("hidden",)
    ln2_scale_logical_axes: tuple = ("hidden",)
    ln2_bias_logical_axes: tuple = ("hidden",)
    final_ln_scale_logical_axes: tuple = ("hidden",)
    final_ln_bias_logical_axes: tuple = ("hidden",)
    qkv_proj_logical_axes: tuple = ("hidden", None)
    out_proj_logical_axes: tuple = ("hidden", None)
    fc1_weight_logical_axes: tuple = ("hidden", None)
    fc1_bias_logical_axes: tuple = (None,)
    fc2_weight_logical_axes: tuple = (None, "hidden")
    fc2_bias_logical_axes: tuple = ("hidden",)
    conv_weight_initializer = None
    cls_token_initializer = None
    position_embedding_initializer = None
    ln1_scale_initializer = None
    ln1_bias_initializer = None
    ln2_scale_initializer = None
    ln2_bias_initializer = None
    final_ln_scale_initializer = None
    final_ln_bias_initializer = None
    qkv_proj_initializer = None
    out_proj_initializer = None
    fc1_weight_initializer = None
    fc1_bias_initializer = None
    fc2_weight_initializer = None
    fc2_bias_initializer = None


@pytest.mark.parametrize(
    "img_size,patch_size,hidden_dim,num_heads,mlp_hidden_dim,num_blocks,cls_flag",
    [
        (64, 8, 192, 3, 768, 2, True),
        (64, 8, 192, 3, 768, 2, False),
    ],
    ids=["with-cls-flag", "without-cls-flag"],
)
def test_vit_init(
    mesh,
    img_size,
    patch_size,
    hidden_dim,
    num_heads,
    mlp_hidden_dim,
    num_blocks,
    cls_flag,
):
    cfg = ViTConfig(
        vit_img_size=img_size,
        vit_patch_size=patch_size,
        vit_hidden_dim=hidden_dim,
        vit_num_heads=num_heads,
        vit_mlp_hidden_dim=mlp_hidden_dim,
        vit_num_blocks=num_blocks,
        vit_cls_flag=cls_flag,
    )

    key = random.PRNGKey(42)
    params = ViT.init(key, mesh, ShardingRule, cfg)

    assert isinstance(params, ViT)
    assert isinstance(params.patch_embedding, PatchEmbeddings)
    assert len(params.blocks) == num_blocks
    assert all(isinstance(block, ViTBlock) for block in params.blocks)
    assert isinstance(params.layer_norm, LayerNorm)
    assert params.cls_flag == cls_flag
    assert params.dropout == cfg.vit_dropout
    assert params.num_blocks == num_blocks


@pytest.mark.parametrize(
    "model_id,revision,num_blocks,hidden_dim,patch_size,img_size,num_patches,cls_flag,batch_size,use_jit",
    [
        (
            "google/siglip-base-patch16-224",
            "main",
            12,
            768,
            16,
            224,
            196,
            False,
            1,
            False,
        ),
        (
            "google/siglip-base-patch16-224",
            "main",
            12,
            768,
            16,
            224,
            196,
            False,
            1,
            True,
        ),
        ("SauravMaheshkar/nanoVLMJAX", "test", 2, 192, 8, 64, 64, True, 1, False),
        ("SauravMaheshkar/nanoVLMJAX", "test", 2, 192, 8, 64, 64, True, 1, True),
    ],
    ids=["siglip-no-jit", "siglip-jit", "hub-no-jit", "hub-jit"],
)
def test_vit_from_pretrained(
    mesh,
    model_id,
    revision,
    num_blocks,
    hidden_dim,
    patch_size,
    img_size,
    num_patches,
    cls_flag,
    batch_size,
    use_jit,
):
    params = ViT.from_pretrained(model_id, revision=revision)

    assert isinstance(params, ViT)
    assert isinstance(params.patch_embedding, PatchEmbeddings)
    assert len(params.blocks) == num_blocks
    assert isinstance(params.layer_norm, LayerNorm)
    assert params.cls_flag is cls_flag

    assert params.patch_embedding.conv_weight.shape == (
        patch_size,
        patch_size,
        3,
        hidden_dim,
    )

    if cls_flag:
        assert params.patch_embedding.position_embedding.shape == (
            1,
            num_patches + 1,
            hidden_dim,
        )
    else:
        assert params.patch_embedding.position_embedding.shape == (
            1,
            num_patches,
            hidden_dim,
        )

    x = random.normal(random.PRNGKey(0), (batch_size, 3, img_size, img_size))

    if use_jit:
        forward_fn = jax.jit(vit_forward)
        output = forward_fn(params, x)
    else:
        output = vit_forward(params, x)

    if cls_flag:
        assert output.shape == (batch_size, hidden_dim)
    else:
        assert output.shape == (batch_size, num_patches, hidden_dim)
    assert output.dtype == jnp.float32


def test_vit_save_pretrained(mesh):
    import json
    import os
    import tempfile

    from safetensors import safe_open

    cfg = ViTConfig(
        vit_img_size=64,
        vit_patch_size=8,
        vit_hidden_dim=192,
        vit_num_heads=3,
        vit_mlp_hidden_dim=768,
        vit_num_blocks=2,
        vit_cls_flag=True,
    )

    key = random.PRNGKey(42)
    params = ViT.init(key, mesh, ShardingRule, cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        params.save_pretrained(tmpdir)

        assert os.path.exists(os.path.join(tmpdir, "model.safetensors"))
        assert os.path.exists(os.path.join(tmpdir, "config.json"))

        with open(os.path.join(tmpdir, "config.json")) as f:
            config = json.load(f)

        assert config["model_type"] == "vit"
        assert config["vit_img_size"] == 64
        assert config["vit_patch_size"] == 8
        assert config["vit_hidden_dim"] == 192
        assert config["vit_num_blocks"] == 2
        assert config["vit_cls_flag"] is True

        weights = {}
        with safe_open(os.path.join(tmpdir, "model.safetensors"), framework="pt") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)

        assert "vision_model.embeddings.patch_embedding.weight" in weights
        assert "vision_model.embeddings.position_embedding.weight" in weights
        assert "vision_model.embeddings.cls_token" in weights
        assert "vision_model.post_layernorm.weight" in weights


def test_vit_save_and_load(mesh):
    import tempfile

    cfg = ViTConfig(
        vit_img_size=64,
        vit_patch_size=8,
        vit_hidden_dim=192,
        vit_num_heads=3,
        vit_mlp_hidden_dim=768,
        vit_num_blocks=2,
        vit_cls_flag=True,
    )

    key = random.PRNGKey(42)
    params = ViT.init(key, mesh, ShardingRule, cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        params.save_pretrained(tmpdir)

        loaded_params = ViT.from_pretrained(tmpdir)

        assert isinstance(loaded_params, ViT)
        assert len(loaded_params.blocks) == params.num_blocks

        assert (
            loaded_params.patch_embedding.conv_weight.shape
            == params.patch_embedding.conv_weight.shape
        )
        assert (
            loaded_params.patch_embedding.position_embedding.shape
            == params.patch_embedding.position_embedding.shape
        )

        loaded_params2 = ViT.from_pretrained(tmpdir)

        assert (
            loaded_params2.patch_embedding.conv_weight.shape
            == params.patch_embedding.conv_weight.shape
        )


@pytest.mark.local
def test_vit_dataloader_forward(mesh):
    import numpy as np
    from transformers import AutoProcessor

    from src.data.datasets import VLMDataset, load_cauldron
    from src.models.vit import ViT, vit_forward

    ds = load_cauldron(
        dataset_path="HuggingFaceM4/the_cauldron",
        dataset_name="tqa",
        max_samples=2,
    )

    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-512")
    dataset = VLMDataset(
        hf_dataset=ds,
        image_processor=processor,
    )

    map_ds = grain.MapDataset.source(dataset)
    map_ds = map_ds.filter(lambda x: len(x.get("messages", [])) > 0)

    assert len(map_ds) > 0, "Filter yielded no elements"

    iter_ds = map_ds.to_iter_dataset(
        grain.ReadOptions(num_threads=1, prefetch_buffer_size=10)
    )

    first_item = next(iter(iter_ds))
    assert first_item is not None
    assert "messages" in first_item

    vit_cfg = ViTConfig(
        vit_img_size=512,
        vit_patch_size=16,
        vit_hidden_dim=768,
        vit_num_heads=12,
        vit_mlp_hidden_dim=3072,
        vit_num_blocks=2,
        vit_cls_flag=True,
    )

    key = random.PRNGKey(42)
    params = ViT.init(key, mesh, ShardingRule, vit_cfg)

    item = dataset[0]
    images = item["images"]
    assert images, "Expected dataset item to contain at least one image"
    if images:
        image = images[0]
        if hasattr(image, "pixel_values"):
            x = image.pixel_values
        else:
            x = image

        x = np.asarray(x)

        # Siglip processor outputs shape (B, C, H, W) where B=1 or (C, H, W)
        # ViT expects exactly (B, C, H, W) with B >= 1
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)  # (C,H,W) -> (B,C,H,W) with B=1
        elif x.ndim == 4 and x.shape[0] == 1:
            pass  # Already (1,C,H,W) - good

        expected_shape = (1, 3, vit_cfg.vit_img_size, vit_cfg.vit_img_size)
        actual_shape = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        assert actual_shape == expected_shape, (
            f"Image shape {actual_shape} doesn't match ViT config "
            f"(B,C,H,W)={expected_shape}"
        )

        output = vit_forward(params, x)

        # With cls_flag=True, ViT returns pooled output (B, hidden_dim)
        # Without cls_flag, returns sequence (B, num_patches, hidden_dim)
        if vit_cfg.vit_cls_flag:
            expected_shape = (1, vit_cfg.vit_hidden_dim)
            assert output.shape == expected_shape, (
                f"Output shape {output.shape} != expected {expected_shape} "
                f"(cls_flag=True gives pooled output)"
            )
        else:
            expected_patches = (vit_cfg.vit_img_size // vit_cfg.vit_patch_size) ** 2
            expected_shape = (1, expected_patches, vit_cfg.vit_hidden_dim)
            assert output.shape == expected_shape, (
                f"Output shape {output.shape} != expected {expected_shape} "
                f"(cls_flag=False gives sequence output)"
            )
