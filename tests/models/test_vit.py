import dataclasses

import jax
import jax.numpy as jnp
import jax.random as random
import pytest

from src.models.vit import (
    LayerNorm,
    Linear,
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
    "img_size,patch_size,hidden_dim,cls_flag",
    [
        (224, 16, 768, True),
        (224, 16, 768, False),
    ],
    ids=["with-cls-flag", "without-cls-flag"],
)
def test_patch_embeddings_init(mesh, img_size, patch_size, hidden_dim, cls_flag):
    cfg = ExampleConfig(
        vit_img_size=img_size,
        vit_patch_size=patch_size,
        vit_hidden_dim=hidden_dim,
        vit_cls_flag=cls_flag,
    )

    key = random.PRNGKey(42)
    params = PatchEmbeddings.init(key, mesh, ShardingRule, cfg)

    assert isinstance(params, PatchEmbeddings)
    assert params.conv_weight.shape == (patch_size, patch_size, 3, hidden_dim)
    assert params.img_size == img_size
    assert params.patch_size == patch_size
    assert params.cls_flag == cls_flag
    assert params.embd_dim == hidden_dim

    num_patches = (img_size // patch_size) ** 2
    if cls_flag:
        assert params.cls_token is not None
        assert params.cls_token.shape == (1, 1, hidden_dim)
        assert params.position_embedding.shape == (1, num_patches + 1, hidden_dim)
    else:
        assert params.cls_token is None
        assert params.position_embedding.shape == (1, num_patches, hidden_dim)


@pytest.mark.parametrize(
    "img_size,patch_size,hidden_dim,cls_flag,batch_size,use_jit",
    [
        (224, 16, 768, True, 2, False),
        (224, 16, 768, False, 2, False),
        (224, 16, 768, True, 2, True),
    ],
    ids=["with-cls-flag", "without-cls-flag", "with-jit"],
)
def test_patch_embeddings_forward(
    mesh, img_size, patch_size, hidden_dim, cls_flag, batch_size, use_jit
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

    if use_jit:
        forward_fn = jax.jit(patch_embeddings_forward)
        output1 = forward_fn(params, x)
        output2 = forward_fn(params, x)
        assert jnp.allclose(output1, output2)
        output = output1
    else:
        output = patch_embeddings_forward(params, x)

    num_patches = (img_size // patch_size) ** 2
    expected_seq_len = num_patches + 1 if cls_flag else num_patches
    assert output.shape == (batch_size, expected_seq_len, hidden_dim)
    assert output.dtype == cfg.dtype


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
    "hidden_dim,num_heads",
    [
        (768, 12),
    ],
    ids=["768-12"],
)
def test_attention_init(mesh, hidden_dim, num_heads):
    cfg = AttentionConfig(vit_hidden_dim=hidden_dim, vit_num_heads=num_heads)

    key = random.PRNGKey(42)
    params = ViTMultiHeadAttention.init(key, mesh, ShardingRule, cfg)

    assert isinstance(params, ViTMultiHeadAttention)
    assert params.qkv_proj.shape == (hidden_dim, 3 * hidden_dim)
    assert params.out_proj.shape == (hidden_dim, hidden_dim)
    assert params.num_heads == num_heads
    assert params.embd_dim == hidden_dim
    assert params.head_dim == hidden_dim // num_heads
    assert params.dropout == cfg.vit_dropout


@pytest.mark.parametrize(
    "hidden_dim,num_heads,batch_size,seq_len,use_jit,use_dropout",
    [
        (768, 12, 2, 197, True, False),
        (768, 12, 2, 197, False, True),
    ],
    ids=["with-jit", "with-dropout"],
)
def test_attention_forward(
    mesh, hidden_dim, num_heads, batch_size, seq_len, use_jit, use_dropout
):
    cfg = AttentionConfig(vit_hidden_dim=hidden_dim, vit_num_heads=num_heads)

    key = random.PRNGKey(42)
    params = ViTMultiHeadAttention.init(key, mesh, ShardingRule, cfg)
    x = random.normal(key, (batch_size, seq_len, hidden_dim))

    if use_dropout:
        key = random.PRNGKey(43)
    else:
        key = None

    if use_jit:
        forward_fn = jax.jit(vit_multi_head_attention_forward)
        output1 = forward_fn(params, x, key)
        output2 = forward_fn(params, x, key)
        assert jnp.allclose(output1, output2)
        output = output1
    else:
        output = vit_multi_head_attention_forward(params, x, key)

    assert output.shape == (batch_size, seq_len, hidden_dim)
    assert output.dtype == cfg.dtype


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
    "hidden_dim,mlp_hidden_dim",
    [
        (192, 768),
    ],
    ids=["192-768"],
)
def test_mlp_init(mesh, hidden_dim, mlp_hidden_dim):
    cfg = MLPConfig(vit_hidden_dim=hidden_dim, vit_mlp_hidden_dim=mlp_hidden_dim)

    key = random.PRNGKey(42)
    params = ViTMLP.init(key, mesh, ShardingRule, cfg)

    assert isinstance(params, ViTMLP)
    assert isinstance(params.fc1, Linear)
    assert isinstance(params.fc2, Linear)
    assert params.fc1.weight.shape == (hidden_dim, mlp_hidden_dim)
    assert params.fc1.bias is not None
    assert params.fc1.bias.shape == (mlp_hidden_dim,)
    assert params.fc2.weight.shape == (mlp_hidden_dim, hidden_dim)
    assert params.fc2.bias is not None
    assert params.fc2.bias.shape == (hidden_dim,)
    assert params.hidden_dim == hidden_dim
    assert params.mlp_hidden_dim == mlp_hidden_dim
    assert params.dropout == cfg.vit_dropout


@pytest.mark.parametrize(
    "hidden_dim,mlp_hidden_dim,batch_size,seq_len,use_jit,use_dropout",
    [
        (768, 3072, 2, 197, True, False),
        (768, 3072, 2, 197, False, True),
    ],
    ids=["with-jit", "with-dropout"],
)
def test_mlp_forward(
    mesh, hidden_dim, mlp_hidden_dim, batch_size, seq_len, use_jit, use_dropout
):
    cfg = MLPConfig(vit_hidden_dim=hidden_dim, vit_mlp_hidden_dim=mlp_hidden_dim)

    key = random.PRNGKey(42)
    params = ViTMLP.init(key, mesh, ShardingRule, cfg)
    x = random.normal(key, (batch_size, seq_len, hidden_dim))

    if use_dropout:
        key = random.PRNGKey(43)
    else:
        key = None

    if use_jit:
        forward_fn = jax.jit(vit_mlp_forward)
        output1 = forward_fn(params, x, key)
        output2 = forward_fn(params, x, key)
        assert jnp.allclose(output1, output2)
        output = output1
    else:
        output = vit_mlp_forward(params, x, key)

    assert output.shape == (batch_size, seq_len, hidden_dim)
    assert output.dtype == cfg.dtype


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
    "normalized_shape,batch_size,seq_len,use_jit",
    [
        (384, 4, 100, False),
        (768, 2, 197, True),
    ],
    ids=["small", "with-jit"],
)
def test_layer_norm_forward(mesh, normalized_shape, batch_size, seq_len, use_jit):
    cfg = LayerNormConfig(normalized_shape=normalized_shape)

    key = random.PRNGKey(42)
    params = LayerNorm.init(key, mesh, ShardingRule, cfg)
    x = random.normal(key, (batch_size, seq_len, normalized_shape))

    if use_jit:
        forward_fn = jax.jit(layer_norm_forward)
        output1 = forward_fn(params, x)
        output2 = forward_fn(params, x)
        assert jnp.allclose(output1, output2)
        output = output1
    else:
        output = layer_norm_forward(params, x)

    assert output.shape == (batch_size, seq_len, normalized_shape)
    assert output.dtype == cfg.dtype


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
    "hidden_dim,num_heads,mlp_hidden_dim",
    [
        (192, 3, 768),
    ],
    ids=["192-3-768"],
)
def test_vit_block_init(mesh, hidden_dim, num_heads, mlp_hidden_dim):
    cfg = ViTBlockConfig(
        vit_hidden_dim=hidden_dim,
        vit_num_heads=num_heads,
        vit_mlp_hidden_dim=mlp_hidden_dim,
    )

    key = random.PRNGKey(42)
    params = ViTBlock.init(key, mesh, ShardingRule, cfg)

    assert isinstance(params, ViTBlock)
    assert isinstance(params.ln1, LayerNorm)
    assert isinstance(params.attn, ViTMultiHeadAttention)
    assert isinstance(params.ln2, LayerNorm)
    assert isinstance(params.mlp, ViTMLP)
    assert params.ln1.normalized_shape == hidden_dim
    assert params.ln2.normalized_shape == hidden_dim
    assert params.attn.embd_dim == hidden_dim
    assert params.attn.num_heads == num_heads
    assert params.mlp.hidden_dim == hidden_dim
    assert params.mlp.mlp_hidden_dim == mlp_hidden_dim


@pytest.mark.parametrize(
    "hidden_dim,num_heads,mlp_hidden_dim,batch_size,seq_len,use_jit,use_dropout",
    [
        (768, 12, 3072, 2, 197, True, False),
        (768, 12, 3072, 2, 197, False, True),
    ],
    ids=["with-jit", "with-dropout"],
)
def test_vit_block_forward(
    mesh,
    hidden_dim,
    num_heads,
    mlp_hidden_dim,
    batch_size,
    seq_len,
    use_jit,
    use_dropout,
):
    cfg = ViTBlockConfig(
        vit_hidden_dim=hidden_dim,
        vit_num_heads=num_heads,
        vit_mlp_hidden_dim=mlp_hidden_dim,
    )

    key = random.PRNGKey(42)
    params = ViTBlock.init(key, mesh, ShardingRule, cfg)
    x = random.normal(key, (batch_size, seq_len, hidden_dim))

    if use_dropout:
        key = random.PRNGKey(43)
    else:
        key = None

    if use_jit:
        forward_fn = jax.jit(vit_block_forward)
        output1 = forward_fn(params, x, key)
        output2 = forward_fn(params, x, key)
        assert jnp.allclose(output1, output2)
        output = output1
    else:
        output = vit_block_forward(params, x, key)

    assert output.shape == (batch_size, seq_len, hidden_dim)
    assert output.dtype == cfg.dtype


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
            2,
            True,
        ),
        ("SauravMaheshkar/nanoVLMJAX", "test", 2, 192, 8, 64, 64, True, 1, False),
        ("SauravMaheshkar/nanoVLMJAX", "test", 2, 192, 8, 64, 64, True, 2, True),
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
        x = x[0]
        x = np.expand_dims(x, axis=0)

        output = vit_forward(params, x)
        assert output.shape[1] == vit_cfg.vit_hidden_dim
