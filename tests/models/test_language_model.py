import dataclasses

import jax
import jax.numpy as jnp
import jax.random as random
import pytest

from src.models.language_model import (
    LanguageModel,
    LanguageModelBlock,
    LanguageModelGroupedQueryAttention,
    LanguageModelMLP,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    language_model_block_forward,
    language_model_forward,
    language_model_gqa_forward,
    language_model_mlp_forward,
    rms_norm_forward,
    rotary_embedding_forward,
)


@dataclasses.dataclass
class LMConfig:
    lm_vocab_size: int = 256
    lm_hidden_dim: int = 64
    lm_n_heads: int = 4
    lm_n_kv_heads: int = 2
    lm_n_layers: int = 2
    lm_inter_dim: int = 128
    lm_dropout: float = 0.1
    lm_rms_eps: float = 1e-6
    lm_re_base: float = 10000.0
    lm_max_position_embeddings: int = 256
    lm_attn_scaling: float = 1.0
    lm_use_tokens: bool = True
    lm_tie_weights: bool = False
    lm_attention_bias: bool = False
    dtype: jnp.dtype = jnp.float32

    token_embedding_logical_axes: tuple = (None, "tp")
    rotary_inv_freq_logical_axes: tuple = (None,)
    rotary_inv_freq_initializer = None
    rms_norm_weight_logical_axes: tuple = (None,)
    rms_norm_weight_initializer = None
    gqa_q_proj_logical_axes: tuple = ("hidden", None)
    gqa_q_proj_initializer = None
    gqa_k_proj_logical_axes: tuple = ("hidden", None)
    gqa_k_proj_initializer = None
    gqa_v_proj_logical_axes: tuple = ("hidden", None)
    gqa_v_proj_initializer = None
    gqa_out_proj_logical_axes: tuple = ("hidden", None)
    gqa_out_proj_initializer = None
    mlp_gate_proj_logical_axes: tuple = ("hidden", None)
    mlp_gate_proj_initializer = None
    mlp_up_proj_logical_axes: tuple = ("hidden", None)
    mlp_up_proj_initializer = None
    mlp_down_proj_logical_axes: tuple = (None, "hidden")
    mlp_down_proj_initializer = None
    block_norm1_weight_logical_axes: tuple = (None,)
    block_norm1_weight_initializer = None
    block_norm2_weight_logical_axes: tuple = (None,)
    block_norm2_weight_initializer = None
    final_norm_weight_logical_axes: tuple = (None,)
    final_norm_weight_initializer = None
    head_logical_axes: tuple = ("hidden", None)
    head_initializer = None
    token_embedding_initializer = None


class ShardingRule:
    batch = "fsdp"
    seq = None
    hidden = "tp"
    linear_in = None
    linear_out = "tp"
    tp = "tp"


#### RMSNorm Tests ####


@pytest.mark.parametrize(
    "batch_size,seq_len,hidden_dim",
    [
        (2, 32, 64),
    ],
    ids=["with-jit"],
)
def test_rms_norm_forward(mesh, batch_size, seq_len, hidden_dim):
    cfg = LMConfig(lm_hidden_dim=hidden_dim)

    key = random.PRNGKey(42)
    params = RMSNorm.init(key, mesh, ShardingRule, cfg)
    x = random.normal(key, (batch_size, seq_len, hidden_dim))

    forward_fn = jax.jit(rms_norm_forward)
    output = forward_fn(params, x)

    assert output.shape == (batch_size, seq_len, hidden_dim)
    assert output.dtype == cfg.dtype


def test_rms_norm_normalization(mesh):
    cfg = LMConfig(lm_hidden_dim=128, lm_rms_eps=1e-6)

    key = random.PRNGKey(42)
    params = RMSNorm.init(key, mesh, ShardingRule, cfg)
    x = jnp.ones((2, 10, 128))

    output = rms_norm_forward(params, x)

    rms = jnp.sqrt(jnp.mean(jnp.square(output), axis=-1))
    assert jnp.allclose(rms, jnp.ones_like(rms), atol=1e-4)


#### Rotary Embedding Tests ####


@pytest.mark.parametrize(
    "batch_size,seq_len,hidden_dim,n_heads",
    [
        (2, 32, 64, 4),
    ],
    ids=["batch-2-seq-32"],
)
def test_rotary_embedding_forward(mesh, batch_size, seq_len, hidden_dim, n_heads):
    cfg = LMConfig(lm_hidden_dim=hidden_dim, lm_n_heads=n_heads)

    key = random.PRNGKey(42)
    params = RotaryEmbedding.init(key, mesh, ShardingRule, cfg)
    position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)

    cos, sin = rotary_embedding_forward(params, position_ids)

    head_dim = hidden_dim // n_heads
    assert cos.shape == (batch_size, seq_len, head_dim)
    assert sin.shape == (batch_size, seq_len, head_dim)
    assert jnp.allclose(cos**2 + sin**2, jnp.ones_like(cos), atol=1e-5)


@pytest.mark.parametrize(
    "batch_size,n_heads,seq_len,head_dim",
    [
        (2, 4, 32, 16),
    ],
    ids=["standard"],
)
def test_apply_rotary_pos_emb(mesh, batch_size, n_heads, seq_len, head_dim):
    q = random.normal(random.PRNGKey(0), (batch_size, n_heads, seq_len, head_dim))
    k = random.normal(random.PRNGKey(1), (batch_size, n_heads, seq_len, head_dim))
    cos = random.normal(random.PRNGKey(2), (batch_size, seq_len, head_dim))
    sin = random.normal(random.PRNGKey(3), (batch_size, seq_len, head_dim))

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


#### Grouped Query Attention Tests ####


@pytest.mark.parametrize(
    "batch_size,seq_len,hidden_dim,n_heads,n_kv_heads",
    [
        (2, 32, 64, 4, 2),
    ],
    ids=["with-jit"],
)
def test_gqa_forward(mesh, batch_size, seq_len, hidden_dim, n_heads, n_kv_heads):
    cfg = LMConfig(
        lm_hidden_dim=hidden_dim,
        lm_n_heads=n_heads,
        lm_n_kv_heads=n_kv_heads,
    )

    key = random.PRNGKey(42)
    params = LanguageModelGroupedQueryAttention.init(key, mesh, ShardingRule, cfg)
    x = random.normal(key, (batch_size, seq_len, hidden_dim))

    position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    rotary_params = RotaryEmbedding.init(key, mesh, ShardingRule, cfg)
    cos, sin = rotary_embedding_forward(rotary_params, position_ids)

    key = random.PRNGKey(43)
    forward_fn = jax.jit(language_model_gqa_forward)
    output1 = forward_fn(params, x, cos, sin, key)
    output2 = forward_fn(params, x, cos, sin, key)
    assert jnp.allclose(output1, output2)

    assert output1.shape == (batch_size, seq_len, hidden_dim)


#### MLP Tests ####


@pytest.mark.parametrize(
    "batch_size,seq_len,hidden_dim,inter_dim",
    [
        (2, 32, 64, 128),
    ],
    ids=["with-jit"],
)
def test_mlp_forward(mesh, batch_size, seq_len, hidden_dim, inter_dim):
    cfg = LMConfig(lm_hidden_dim=hidden_dim, lm_inter_dim=inter_dim)

    key = random.PRNGKey(42)
    params = LanguageModelMLP.init(key, mesh, ShardingRule, cfg)
    x = random.normal(key, (batch_size, seq_len, hidden_dim))

    forward_fn = jax.jit(language_model_mlp_forward)
    output = forward_fn(params, x)

    assert output.shape == (batch_size, seq_len, hidden_dim)


#### Language Model Block Tests ####


@pytest.mark.parametrize(
    "batch_size,seq_len,hidden_dim,n_heads,n_kv_heads,inter_dim",
    [
        (2, 32, 64, 4, 2, 128),
    ],
    ids=["with-jit"],
)
def test_block_forward(
    mesh,
    batch_size,
    seq_len,
    hidden_dim,
    n_heads,
    n_kv_heads,
    inter_dim,
):
    cfg = LMConfig(
        lm_hidden_dim=hidden_dim,
        lm_n_heads=n_heads,
        lm_n_kv_heads=n_kv_heads,
        lm_inter_dim=inter_dim,
    )

    key = random.PRNGKey(42)
    params = LanguageModelBlock.init(key, mesh, ShardingRule, cfg)
    x = random.normal(key, (batch_size, seq_len, hidden_dim))

    position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    rotary_params = RotaryEmbedding.init(key, mesh, ShardingRule, cfg)
    cos, sin = rotary_embedding_forward(rotary_params, position_ids)

    forward_fn = jax.jit(language_model_block_forward)
    output = forward_fn(params, x, cos, sin)

    assert output.shape == (batch_size, seq_len, hidden_dim)


#### Language Model Tests ####


@pytest.mark.parametrize(
    "batch_size,seq_len,vocab_size,hidden_dim,n_heads,n_kv_heads,n_layers,inter_dim",
    [
        (2, 32, 256, 64, 4, 2, 2, 128),
    ],
    ids=["with-jit"],
)
def test_language_model_forward(
    mesh,
    batch_size,
    seq_len,
    vocab_size,
    hidden_dim,
    n_heads,
    n_kv_heads,
    n_layers,
    inter_dim,
):
    cfg = LMConfig(
        lm_vocab_size=vocab_size,
        lm_hidden_dim=hidden_dim,
        lm_n_heads=n_heads,
        lm_n_kv_heads=n_kv_heads,
        lm_n_layers=n_layers,
        lm_inter_dim=inter_dim,
    )

    key = random.PRNGKey(42)
    params = LanguageModel.init(key, mesh, ShardingRule, cfg)
    input_ids = random.randint(key, (batch_size, seq_len), 0, vocab_size)

    forward_fn = jax.jit(language_model_forward)
    output = forward_fn(params, input_ids)

    assert output.shape == (batch_size, seq_len, vocab_size)


def test_language_model_attention_mask(mesh):
    cfg = LMConfig(lm_hidden_dim=64, lm_n_heads=4, lm_n_kv_heads=2)

    key = random.PRNGKey(42)
    params = LanguageModel.init(key, mesh, ShardingRule, cfg)

    batch_size, seq_len = 2, 32
    input_ids = random.randint(key, (batch_size, seq_len), 0, cfg.lm_vocab_size)

    attention_mask = jnp.ones((batch_size, seq_len))
    attention_mask = attention_mask.at[:, 16:].set(0)

    output = language_model_forward(params, input_ids, attention_mask=attention_mask)

    assert output.shape == (batch_size, seq_len, cfg.lm_vocab_size)


@pytest.mark.parametrize(
    "model_id,revision,vocab_size,hidden_dim,n_heads,n_kv_heads,n_layers,inter_dim,batch_size,seq_len,use_jit",
    [
        (
            "HuggingFaceTB/SmolLM2-135M",
            "main",
            49152,
            576,
            9,
            3,
            30,
            1536,
            1,
            32,
            True,
        ),
    ],
    ids=["smollm2-135m"],
)
def test_language_model_from_pretrained(
    mesh,
    model_id,
    revision,
    vocab_size,
    hidden_dim,
    n_heads,
    n_kv_heads,
    n_layers,
    inter_dim,
    batch_size,
    seq_len,
    use_jit,
):
    params = LanguageModel.from_pretrained(model_id, revision=revision)

    assert isinstance(params, LanguageModel)
    assert params.token_embedding.shape == (vocab_size, hidden_dim)
    assert len(params.blocks) == n_layers
    assert hasattr(params, "norm")
    assert params.head.shape == (hidden_dim, vocab_size)

    input_ids = random.randint(random.PRNGKey(0), (batch_size, seq_len), 0, vocab_size)

    if use_jit:
        forward_fn = jax.jit(language_model_forward)
        output = forward_fn(params, input_ids)
    else:
        output = language_model_forward(params, input_ids)

    assert output.shape == (batch_size, seq_len, vocab_size)


def test_language_model_save_pretrained(mesh):
    import tempfile

    cfg = LMConfig(
        lm_vocab_size=256,
        lm_hidden_dim=64,
        lm_n_heads=4,
        lm_n_kv_heads=2,
        lm_n_layers=2,
        lm_inter_dim=128,
    )

    key = random.PRNGKey(42)
    params = LanguageModel.init(key, mesh, ShardingRule, cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        params.save_pretrained(tmpdir)

        loaded_params = LanguageModel.from_pretrained(tmpdir)

        assert isinstance(loaded_params, LanguageModel)
        assert len(loaded_params.blocks) == params.num_layers

        assert loaded_params.token_embedding.shape == params.token_embedding.shape
        assert loaded_params.head.shape == params.head.shape

        loaded_params2 = LanguageModel.from_pretrained(tmpdir)

        assert loaded_params2.token_embedding.shape == params.token_embedding.shape
        assert loaded_params2.head.shape == params.head.shape

        input_ids = random.randint(
            random.PRNGKey(0),
            (1, 4),
            0,
            cfg.lm_vocab_size,
        )
        output = language_model_forward(loaded_params, input_ids)
        assert output.shape == (1, 4, cfg.lm_vocab_size)
