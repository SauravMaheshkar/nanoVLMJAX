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
    rotate_half,
)


@dataclasses.dataclass
class LMConfig:
    lm_vocab_size: int = 32000
    lm_hidden_dim: int = 768
    lm_n_heads: int = 12
    lm_n_kv_heads: int = 4
    lm_n_layers: int = 2
    lm_inter_dim: int = 2048
    lm_dropout: float = 0.1
    lm_rms_eps: float = 1e-6
    lm_re_base: float = 10000.0
    lm_max_position_embeddings: int = 2048
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
    "hidden_dim,eps",
    [
        (768, 1e-6),
        (1024, 1e-5),
    ],
    ids=["768-1e-6", "1024-1e-5"],
)
def test_rms_norm_init(mesh, hidden_dim, eps):
    cfg = LMConfig(lm_hidden_dim=hidden_dim, lm_rms_eps=eps)

    key = random.PRNGKey(42)
    params = RMSNorm.init(key, mesh, ShardingRule, cfg)

    assert isinstance(params, RMSNorm)
    assert params.weight.shape == (hidden_dim,)
    assert params.normalized_shape == hidden_dim
    assert params.eps == eps


@pytest.mark.parametrize(
    "batch_size,seq_len,hidden_dim,use_jit",
    [
        (2, 128, 768, True),
        (2, 128, 768, False),
    ],
    ids=["with-jit", "without-jit"],
)
def test_rms_norm_forward(mesh, batch_size, seq_len, hidden_dim, use_jit):
    cfg = LMConfig(lm_hidden_dim=hidden_dim)

    key = random.PRNGKey(42)
    params = RMSNorm.init(key, mesh, ShardingRule, cfg)
    x = random.normal(key, (batch_size, seq_len, hidden_dim))

    if use_jit:
        forward_fn = jax.jit(rms_norm_forward)
        output = forward_fn(params, x)
    else:
        output = rms_norm_forward(params, x)

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


def test_rotate_half(mesh):
    x = jnp.arange(8).reshape(1, 4, 2).astype(jnp.float32)
    result = rotate_half(x)
    expected = jnp.concatenate([-x[..., 1:], x[..., :1]], axis=-1)
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize(
    "hidden_dim,n_heads",
    [(768, 12)],
    ids=["768-12"],
)
def test_rotary_embedding_init(mesh, hidden_dim, n_heads):
    cfg = LMConfig(lm_hidden_dim=hidden_dim, lm_n_heads=n_heads)

    key = random.PRNGKey(42)
    params = RotaryEmbedding.init(key, mesh, ShardingRule, cfg)

    assert isinstance(params, RotaryEmbedding)
    assert params.head_dim == hidden_dim // n_heads
    assert params.base == cfg.lm_re_base
    assert params.max_seq_len == cfg.lm_max_position_embeddings
    assert params.attention_scaling == cfg.lm_attn_scaling


@pytest.mark.parametrize(
    "batch_size,seq_len,hidden_dim,n_heads",
    [
        (2, 128, 768, 12),
    ],
    ids=["batch-2-seq-128"],
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
        (2, 12, 128, 64),
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
    "hidden_dim,n_heads,n_kv_heads",
    [(768, 12, 4)],
    ids=["standard-gqa"],
)
def test_gqa_init(mesh, hidden_dim, n_heads, n_kv_heads):
    cfg = LMConfig(
        lm_hidden_dim=hidden_dim,
        lm_n_heads=n_heads,
        lm_n_kv_heads=n_kv_heads,
    )

    key = random.PRNGKey(42)
    params = LanguageModelGroupedQueryAttention.init(key, mesh, ShardingRule, cfg)

    assert isinstance(params, LanguageModelGroupedQueryAttention)
    assert params.q_proj.shape == (hidden_dim, hidden_dim)
    assert params.k_proj.shape == (hidden_dim, (hidden_dim // n_heads) * n_kv_heads)
    assert params.v_proj.shape == (hidden_dim, (hidden_dim // n_heads) * n_kv_heads)
    assert params.out_proj.shape == (hidden_dim, hidden_dim)
    assert params.n_heads == n_heads
    assert params.n_kv_heads == n_kv_heads
    assert params.head_dim == hidden_dim // n_heads


@pytest.mark.parametrize(
    "batch_size,seq_len,hidden_dim,n_heads,n_kv_heads,use_jit,use_dropout",
    [
        (2, 128, 768, 12, 4, True, False),
        (2, 128, 768, 12, 4, False, True),
    ],
    ids=["with-jit", "with-dropout"],
)
def test_gqa_forward(
    mesh, batch_size, seq_len, hidden_dim, n_heads, n_kv_heads, use_jit, use_dropout
):
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

    if use_dropout:
        key = random.PRNGKey(43)
    else:
        key = None

    if use_jit:
        forward_fn = jax.jit(language_model_gqa_forward)
        output1 = forward_fn(params, x, cos, sin, key)
        output2 = forward_fn(params, x, cos, sin, key)
        assert jnp.allclose(output1, output2)
        output = output1
    else:
        output = language_model_gqa_forward(params, x, cos, sin, key)

    assert output.shape == (batch_size, seq_len, hidden_dim)


#### MLP Tests ####


@pytest.mark.parametrize(
    "hidden_dim,inter_dim",
    [(768, 2048)],
    ids=["standard-mlp"],
)
def test_mlp_init(mesh, hidden_dim, inter_dim):
    cfg = LMConfig(lm_hidden_dim=hidden_dim, lm_inter_dim=inter_dim)

    key = random.PRNGKey(42)
    params = LanguageModelMLP.init(key, mesh, ShardingRule, cfg)

    assert isinstance(params, LanguageModelMLP)
    assert params.gate_proj.shape == (hidden_dim, inter_dim)
    assert params.up_proj.shape == (hidden_dim, inter_dim)
    assert params.down_proj.shape == (inter_dim, hidden_dim)
    assert params.hidden_dim == hidden_dim
    assert params.inter_dim == inter_dim


@pytest.mark.parametrize(
    "batch_size,seq_len,hidden_dim,inter_dim,use_jit",
    [
        (2, 128, 768, 2048, True),
        (2, 128, 768, 2048, False),
    ],
    ids=["with-jit", "without-jit"],
)
def test_mlp_forward(mesh, batch_size, seq_len, hidden_dim, inter_dim, use_jit):
    cfg = LMConfig(lm_hidden_dim=hidden_dim, lm_inter_dim=inter_dim)

    key = random.PRNGKey(42)
    params = LanguageModelMLP.init(key, mesh, ShardingRule, cfg)
    x = random.normal(key, (batch_size, seq_len, hidden_dim))

    if use_jit:
        forward_fn = jax.jit(language_model_mlp_forward)
        output = forward_fn(params, x)
    else:
        output = language_model_mlp_forward(params, x)

    assert output.shape == (batch_size, seq_len, hidden_dim)


#### Language Model Block Tests ####


@pytest.mark.parametrize(
    "hidden_dim,n_heads,n_kv_heads,n_layers,inter_dim",
    [(768, 12, 4, 2, 2048)],
    ids=["standard-block"],
)
def test_block_init(mesh, hidden_dim, n_heads, n_kv_heads, n_layers, inter_dim):
    cfg = LMConfig(
        lm_hidden_dim=hidden_dim,
        lm_n_heads=n_heads,
        lm_n_kv_heads=n_kv_heads,
        lm_n_layers=n_layers,
        lm_inter_dim=inter_dim,
    )

    key = random.PRNGKey(42)
    params = LanguageModelBlock.init(key, mesh, ShardingRule, cfg)

    assert isinstance(params, LanguageModelBlock)
    assert hasattr(params, "norm1")
    assert hasattr(params, "norm2")
    assert hasattr(params, "attn")
    assert hasattr(params, "mlp")


@pytest.mark.parametrize(
    "batch_size,seq_len,hidden_dim,n_heads,n_kv_heads,inter_dim,use_jit",
    [
        (2, 128, 768, 12, 4, 2048, True),
        (2, 128, 768, 12, 4, 2048, False),
    ],
    ids=["with-jit", "without-jit"],
)
def test_block_forward(
    mesh,
    batch_size,
    seq_len,
    hidden_dim,
    n_heads,
    n_kv_heads,
    inter_dim,
    use_jit,
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

    if use_jit:
        forward_fn = jax.jit(language_model_block_forward)
        output = forward_fn(params, x, cos, sin)
    else:
        output = language_model_block_forward(params, x, cos, sin)

    assert output.shape == (batch_size, seq_len, hidden_dim)


#### Language Model Tests ####


@pytest.mark.parametrize(
    "vocab_size,hidden_dim,n_heads,n_kv_heads,n_layers,inter_dim",
    [(32000, 768, 12, 4, 2, 2048)],
    ids=["standard-lm"],
)
def test_language_model_init(
    mesh, vocab_size, hidden_dim, n_heads, n_kv_heads, n_layers, inter_dim
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

    assert isinstance(params, LanguageModel)
    assert params.token_embedding.shape == (vocab_size, hidden_dim)
    assert hasattr(params, "rotary_emb")
    assert len(params.blocks) == n_layers
    assert hasattr(params, "norm")
    assert params.head.shape == (hidden_dim, vocab_size)
    assert params.vocab_size == vocab_size
    assert params.hidden_dim == hidden_dim
    assert params.num_layers == n_layers


@pytest.mark.parametrize(
    "batch_size,seq_len,vocab_size,hidden_dim,n_heads,n_kv_heads,n_layers,inter_dim,use_jit",
    [
        (2, 128, 32000, 768, 12, 4, 2, 2048, True),
        (2, 128, 32000, 768, 12, 4, 2, 2048, False),
    ],
    ids=["with-jit", "without-jit"],
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
    use_jit,
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

    if use_jit:
        forward_fn = jax.jit(language_model_forward)
        output = forward_fn(params, input_ids)
    else:
        output = language_model_forward(params, input_ids)

    assert output.shape == (batch_size, seq_len, vocab_size)


def test_language_model_attention_mask(mesh):
    cfg = LMConfig(lm_hidden_dim=768, lm_n_heads=12, lm_n_kv_heads=4)

    key = random.PRNGKey(42)
    params = LanguageModel.init(key, mesh, ShardingRule, cfg)

    batch_size, seq_len = 2, 128
    input_ids = random.randint(key, (batch_size, seq_len), 0, cfg.lm_vocab_size)

    attention_mask = jnp.ones((batch_size, seq_len))
    attention_mask = attention_mask.at[:, 64:].set(0)

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
        lm_vocab_size=32000,
        lm_hidden_dim=576,
        lm_n_heads=9,
        lm_n_kv_heads=3,
        lm_n_layers=2,
        lm_inter_dim=1536,
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
