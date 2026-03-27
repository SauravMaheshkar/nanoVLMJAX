import dataclasses

import jax
import jax.numpy as jnp
import jax.random as random
import pytest

from src.models.vision_language_model import (
    VisionLanguageModel,
    vlm_forward,
)


@dataclasses.dataclass
class VLMTestConfig:
    vit_hidden_dim: int = 64
    vit_inter_dim: int = 128
    vit_patch_size: int = 16
    vit_img_size: int = 32
    vit_n_heads: int = 4
    vit_dropout: float = 0.0
    vit_n_blocks: int = 2
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False

    lm_hidden_dim: int = 64
    lm_inter_dim: int = 128
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 256
    lm_vocab_size: int = 256
    lm_n_heads: int = 4
    lm_n_kv_heads: int = 2
    lm_dropout: float = 0.0
    lm_n_layers: int = 2
    lm_attn_scaling: float = 1.0
    lm_use_tokens: bool = False
    lm_tie_weights: bool = True
    lm_attention_bias: bool = False

    mp_pixel_shuffle_factor: int = 2
    mp_image_token_length: int = 64

    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = "nanoVLM"
    image_token_id: int = 0  # Set to tokenizer.image_token_id

    dtype: jnp.dtype = jnp.float32


class ShardingRule:
    batch = "fsdp"
    seq = None
    hidden = "tp"
    tp = "tp"
    kernel_h = None
    kernel_w = None
    in_channels = None
    out_channels = "tp"


#### VisionLanguageModel Tests ####


@pytest.mark.parametrize(
    "vit_hidden_dim,lm_hidden_dim",
    [(768, 576)],
    ids=["standard"],
)
def test_vlm_init(mesh, vit_hidden_dim, lm_hidden_dim):
    cfg = VLMTestConfig(
        vit_hidden_dim=vit_hidden_dim,
        lm_hidden_dim=lm_hidden_dim,
    )

    key = random.PRNGKey(42)
    params = VisionLanguageModel.init(key, mesh, ShardingRule, cfg)

    assert isinstance(params, VisionLanguageModel)
    assert hasattr(params, "vision_encoder")
    assert hasattr(params, "modality_projector")
    assert hasattr(params, "decoder")


@pytest.mark.parametrize(
    "batch_size,seq_len,vocab_size,use_jit",
    [
        (1, 16, 256, True),
        (1, 16, 256, False),
    ],
    ids=["with-jit", "without-jit"],
)
def test_vlm_forward(
    mesh,
    batch_size,
    seq_len,
    vocab_size,
    use_jit,
):
    cfg = VLMTestConfig(
        lm_vocab_size=vocab_size,
    )

    key = random.PRNGKey(42)
    params = VisionLanguageModel.init(key, mesh, ShardingRule, cfg)
    input_ids = random.randint(key, (batch_size, seq_len), 0, vocab_size)

    if use_jit:
        forward_fn = jax.jit(vlm_forward)
        output = forward_fn(params, input_ids, images=None)
    else:
        output = vlm_forward(params, input_ids, images=None)

    assert output.shape == (batch_size, seq_len, vocab_size)


@pytest.mark.parametrize(
    "batch_size,seq_len,vocab_size,use_jit",
    [
        (1, 16, 256, True),
        (1, 16, 256, False),
    ],
    ids=["multimodal-with-jit", "multimodal-without-jit"],
)
def test_vlm_forward_multimodal(
    mesh,
    batch_size,
    seq_len,
    vocab_size,
    use_jit,
):
    cfg = VLMTestConfig(
        lm_vocab_size=vocab_size,
        vit_img_size=32,
    )

    key = random.PRNGKey(42)
    params = VisionLanguageModel.init(key, mesh, ShardingRule, cfg)

    input_ids = random.randint(key, (batch_size, seq_len), 0, vocab_size)
    input_ids = input_ids.at[:, 0].set(cfg.image_token_id)

    images = random.normal(key, (batch_size, 3, cfg.vit_img_size, cfg.vit_img_size))

    if use_jit:
        forward_fn = jax.jit(vlm_forward)
        output = forward_fn(params, input_ids, images=images)
    else:
        output = vlm_forward(params, input_ids, images=images)

    assert output.shape == (batch_size, seq_len, vocab_size)


@pytest.mark.parametrize(
    "model_id,revision,vocab_size,hidden_dim,vit_hidden_dim,n_heads,n_kv_heads,n_layers,inter_dim,batch_size,seq_len,use_jit",
    [
        (
            "ariG23498/nanoVLM-demo",
            "main",
            49152,
            576,
            768,
            9,
            3,
            30,
            1536,
            1,
            32,
            True,
        ),
    ],
    ids=["nanoVLM-demo"],
)
def test_vlm_from_pretrained(
    mesh,
    model_id,
    revision,
    vocab_size,
    hidden_dim,
    vit_hidden_dim,
    n_heads,
    n_kv_heads,
    n_layers,
    inter_dim,
    batch_size,
    seq_len,
    use_jit,
):
    params = VisionLanguageModel.from_pretrained(model_id, revision=revision)

    assert isinstance(params, VisionLanguageModel)
    assert params.decoder.token_embedding.shape == (vocab_size, hidden_dim)
    assert len(params.decoder.blocks) == n_layers

    input_ids = random.randint(random.PRNGKey(0), (batch_size, seq_len), 0, vocab_size)

    if use_jit:
        forward_fn = jax.jit(vlm_forward)
        output = forward_fn(params, input_ids, images=None)
    else:
        output = vlm_forward(params, input_ids, images=None)

    assert output.shape == (batch_size, seq_len, vocab_size)


def test_vlm_save_pretrained(mesh):
    import tempfile

    cfg = VLMTestConfig(
        lm_vocab_size=256,
        lm_hidden_dim=64,
        lm_n_heads=4,
        lm_n_kv_heads=2,
        lm_n_layers=2,
        lm_inter_dim=128,
        vit_hidden_dim=64,
        vit_n_heads=4,
        vit_n_blocks=2,
    )

    key = random.PRNGKey(42)
    params = VisionLanguageModel.init(key, mesh, ShardingRule, cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        params.save_pretrained(tmpdir)

        loaded_params = VisionLanguageModel.from_pretrained(tmpdir)

        assert isinstance(loaded_params, VisionLanguageModel)
        assert len(loaded_params.decoder.blocks) == params.decoder.num_layers

        assert (
            loaded_params.decoder.token_embedding.shape
            == params.decoder.token_embedding.shape
        )
