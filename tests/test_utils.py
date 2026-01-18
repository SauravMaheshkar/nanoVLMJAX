import dataclasses
from typing import Self

import jax
import jax.nn.initializers
import jax.numpy as jnp
import jax.random as random
import pytest
from jax import tree_util as jtu
from jax.sharding import PartitionSpec

from src.utils import (
    ParamInitializer,
    ParamSpec,
    _init_leaves,
    jax_pytree_struct,
    logical_to_physical,
    logical_to_sharding,
)


def test_paramspec() -> None:
    # register the paramspec dataclass as a jax pytree
    RegisteredParamSpec = jax_pytree_struct(ParamSpec)

    # dummy instance
    spec = RegisteredParamSpec(
        shape=(10, 20),
        logical_axes=("batch", "feature"),
        initializer=None,
        dtype=jnp.float32,
    )

    # flatten - only data fields should appear as leaves
    leaves, treedef = jtu.tree_flatten(spec)

    # verify that only dtype (a data field) is in the leaves
    assert len(leaves) == 1
    assert leaves[0] == jnp.float32

    # verify that meta fields are preserved in the structure
    # i.e. when unflattening, the meta fields should be restored
    reconstructed = jtu.tree_unflatten(treedef, leaves)

    # verify all fields are preserved
    assert reconstructed.shape == (10, 20)
    assert reconstructed.logical_axes == ("batch", "feature")
    assert reconstructed.initializer is None
    assert reconstructed.dtype == jnp.float32


def illustrate_non_frozen_dataclass_fails_jax_purity() -> None:
    # dataclass without frozen=True
    @dataclasses.dataclass
    class MutableSpec:
        shape: tuple[int, ...] = dataclasses.field(
            metadata=dict(static=True)
        )  # meta_field
        value: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(1.0))

    # register the mutable dataclass as a jax pytree
    RegisteredMutableSpec = jax_pytree_struct(MutableSpec)

    # dummy instance
    spec = RegisteredMutableSpec(shape=(10, 20), value=jnp.array(2.0))

    # basic pytree operations still work
    leaves, _ = jtu.tree_flatten(spec)

    # only value the data field appears as leaf, shape is meta_field
    assert len(leaves) == 1
    assert leaves[0] == jnp.array(2.0)

    # however, non-frozen dataclasses are not hashable, which violates
    # hashability ! essential for cache key generation in jit compilation
    with pytest.raises(TypeError, match="unhashable type"):
        hash(spec)

    # even with only hashable fields non-frozen dataclasses cannot be hashed
    @dataclasses.dataclass
    class MutableSpecHashableFields:
        shape: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))
        name: str = dataclasses.field(default="test")

    RegisteredMutableSpecHashable = jax_pytree_struct(MutableSpecHashableFields)
    mutable_spec_hashable = RegisteredMutableSpecHashable(shape=(10, 20), name="test")

    with pytest.raises(TypeError, match="unhashable type"):
        hash(mutable_spec_hashable)


class LinearShardingRule:
    """Illustrative sharding rules for linear layers.

    |           logical axes          |      physical mesh axes    |
    |---------------------------------|----------------------------|
    |   in_features (input dimension) |  None or model parallelism |
    | out_features (output dimension) |     model parallelism      |
    |     batch (batch dimension)     |      data parallelism      |
    |      seq (sequence length)      | typically unsharded (None) |
    """

    in_features = None
    out_features = "tp"
    batch = "fsdp"
    seq = None


class CollidingRule:
    in_features = "tp"
    out_features = "tp"


@pytest.mark.parametrize(
    "sharding_rule,logical_axes,expected_physical_axes,error",
    [
        (
            LinearShardingRule,
            ("in_features", "out_features"),
            PartitionSpec(None, "tp"),
            None,
        ),
        (
            LinearShardingRule,
            ("batch", None, "out_features"),
            PartitionSpec("fsdp", None, "tp"),
            None,
        ),
        (LinearShardingRule, (None, None, None), PartitionSpec(None, None, None), None),
        (LinearShardingRule, (), PartitionSpec(), None),
        (
            CollidingRule,
            ("in_features", "out_features"),
            None,
            "Colliding physical axes",
        ),
    ],
)
def test_sharding_utilities(
    sharding_rule, logical_axes, expected_physical_axes, error, mesh
) -> None:
    if error is not None:
        with pytest.raises(ValueError, match=error):
            logical_to_physical(logical_axes=logical_axes, rules=sharding_rule)
    else:
        physical = logical_to_physical(logical_axes=logical_axes, rules=sharding_rule)
        assert physical == expected_physical_axes

        sharding = logical_to_sharding(
            logical_axes=logical_axes, mesh=mesh, rules=sharding_rule
        )
        assert sharding.spec == expected_physical_axes


def test_init_leaves(mesh):
    class ShardingRule:
        batch = "fsdp"
        feature = "tp"

    spec = ParamSpec(
        shape=(4, 8),
        logical_axes=("batch", "feature"),
        initializer=jax.nn.initializers.normal(),
        dtype=jnp.float32,
    )

    param_specs = (spec,)
    sharding = logical_to_sharding(
        logical_axes=("batch", "feature"), mesh=mesh, rules=ShardingRule
    )
    shardings = (sharding,)

    key1 = random.PRNGKey(42)  # answer to everything
    key2 = random.PRNGKey(73)  # best number as per sheldon cooper

    arr1 = _init_leaves(key1, param_specs, shardings)[0]
    arr2 = _init_leaves(key2, param_specs, shardings)[0]

    assert not jnp.allclose(arr1, arr2)


@jax_pytree_struct
@dataclasses.dataclass
class SimpleLayer(ParamInitializer):
    size: int = dataclasses.field(metadata=dict(static=True))
    weight: ParamSpec | jnp.ndarray
    bias: ParamSpec | jnp.ndarray

    @classmethod
    def param_specs(cls, size, dtype=jnp.float32) -> Self:
        weight = ParamSpec(
            shape=(size, size),
            logical_axes=("in_features", "out_features"),
            initializer=jax.nn.initializers.normal(),
            dtype=dtype,
        )
        bias = ParamSpec(
            shape=(size,),
            logical_axes=("out_features",),
            initializer=jax.nn.initializers.zeros,
            dtype=dtype,
        )
        return cls(weight=weight, bias=bias, size=size)


def test_param_initializer(mesh):
    class ShardingRule:
        in_features = None
        out_features = "tp"

    key = random.PRNGKey(42)
    size = 8

    shardings = SimpleLayer.get_sharding(mesh, ShardingRule, size)

    assert isinstance(shardings, SimpleLayer)
    assert shardings.weight.spec == PartitionSpec(None, "tp")
    assert shardings.bias.spec == PartitionSpec("tp")
    assert shardings.size == size

    # initialize !
    params = SimpleLayer.init(key, mesh, ShardingRule, size)

    assert isinstance(params, SimpleLayer)
    assert params.weight.shape == (size, size)
    assert params.weight.dtype == jnp.float32
    assert params.weight.sharding.spec == PartitionSpec(None, "tp")
    assert params.bias.shape == (size,)
    assert params.bias.dtype == jnp.float32
    assert params.bias.sharding.spec == PartitionSpec("tp")
    assert jnp.allclose(params.bias, jnp.zeros(size, dtype=jnp.float32))
    assert params.size == size
