import dataclasses
from functools import partial
from typing import Any, Callable, Tuple, Type

import jax
import jax.numpy as jnp
import jax.random as random
from jax import tree_util as jtu
from jax.sharding import NamedSharding, PartitionSpec

#### Common Types ####
AxisName = str | Tuple[str, ...] | None
Axes = Tuple[AxisName, ...]


#### Utility Functions ####


def istype(x: Any, cls: Type[Any]) -> bool:
    return (type(x).__name__ == cls.__name__) and (type(x).__module__ == cls.__module__)


def jax_pytree_struct(cls):
    """
    A decorator that registers a dataclass as a JAX PyTree, automatically
    detecting static fields from metadata. A field is a class variable with
    a type annotation.

    Any field that is marked with `metadata={'static': True}` is considered
    as a meta_field (non-trainable). These fields must be static, hashable
    and immutable objects, as these objects are used to generate cache keys
    during JIT compilation, i.e. they cannot contain jax.Array or numpy.ndarray
    objects. All other fields are data_fields (trainable) these must contain
    JAX-compatible objects such as arrays or scalars.

    References:
        * https://github.com/AakashKumarNain/nanoGPTJAX/blob/main/nanogpt/utils.py
        * https://github.com/jax-ml/jax-llm-examples/blob/main/qwen3/qwen3_jax/model.py
    """
    # ensure is a dataclass before registering as a JAX PyTree
    if not dataclasses.is_dataclass(cls):
        cls = dataclasses.dataclass(cls)

    # get all fields that are part of the constructor's __init__ method
    all_fields = tuple(f for f in dataclasses.fields(cls) if f.init)

    # partition into meta and data fields
    meta_fields = tuple(f.name for f in all_fields if f.metadata.get("static", False))
    data_fields = tuple(
        f.name for f in all_fields if not f.metadata.get("static", False)
    )

    # extends the set of types that are considered internal to pytrees
    return jtu.register_dataclass(cls, data_fields=data_fields, meta_fields=meta_fields)


def logical_to_physical(logical_axes: Axes, rules) -> jax.sharding.PartitionSpec:
    """
    Returns a jax.sharding.PartitionSpec describing how to partition an array across a
    mesh of devices based on the given logical array dimensions
    (i.e. the logical shape of an array)

    Args:
        logical_axes (Axes): The logical axes to shard.
        rules: The sharding rules to use.

    Returns:
        PartitionSpec: The physical mesh axes to use.

    Example:
        >>> class ShardingRule:
        ...     batch = "data"
        ...     feature = "model"
        >>> logical_to_physical(logical_axes=("batch", "feature"), ShardingRule)
        PartitionSpec("data", "model")

    References:
        * https://docs.jax.dev/en/latest/jax.sharding.html
        * https://github.com/AakashKumarNain/nanoGPTJAX/blob/main/nanogpt/utils.py
        * https://github.com/jax-ml/jax-llm-examples/blob/main/qwen3/qwen3_jax/model.py
    """

    physical_axes = [
        getattr(rules, axis) if axis is not None else None for axis in logical_axes
    ]

    # `physical_axes` may contain tuples, flatten to check that `physical_axes`
    # maps each physical mesh axis to at most one logical array axis.
    flat_axes = jax.tree.leaves(physical_axes)
    if len(set(flat_axes)) != len(flat_axes):
        raise ValueError(
            f"Colliding physical axes from translating logical spec {logical_axes} -> {physical_axes}"  # noqa: E501
        )
    return PartitionSpec(*physical_axes)


def logical_to_sharding(
    logical_axes: Axes, mesh: jax.sharding.Mesh, rules
) -> jax.sharding.NamedSharding:
    """
    Constructs a jax.sharding.NamedSharding object based on the given logical
    array dimensions, mesh of devices and sharding rules.

    Args:
        logical_axes (Axes): The logical axes to shard.
        mesh (jax.sharding.Mesh): The mesh of devices to shard across.
        rules: The sharding rules to use.

    Returns:
        jax.sharding.NamedSharding

    Example:
        >>> class ShardingRule:
        ...     batch = "fsdp"
        ...     feature = "tp"
        >>> mesh = jax.sharding.Mesh(
        ...     devices=np.array(jax.devices()).reshape(2, 4), axis_names=("fsdp", "tp")
        ... )
        >>> sharding = logical_to_sharding(logical_axes=("batch", "feature"),
        ...     mesh=mesh, ShardingRule)
        >>> sharding.spec
        PartitionSpec('fsdp', 'tp')

    References:
        * https://docs.jax.dev/en/latest/jax.sharding.html
        * https://github.com/AakashKumarNain/nanoGPTJAX/blob/main/nanogpt/utils.py
        * https://github.com/jax-ml/jax-llm-examples/blob/main/qwen3/qwen3_jax/model.py
    """

    assert mesh is not None
    return NamedSharding(mesh, logical_to_physical(logical_axes, rules))


@partial(jax.jit, static_argnames=("param_specs", "shardings"))
def _init_leaves(key: jax.random.PRNGKey, param_specs, shardings):
    """
    Initializes a PyTree of parameter arrays from ParamSpec objects using a given key.

    Takes a PyTree of parameter specifications and creates initialized JAX arrays
    with proper sharding across devices.

    Args:
        key: JAX PRNG key for random number generation
        param_specs: PyTree of ParamSpec objects defining parameter specifications
        shardings: PyTree of NamedSharding objects

    Returns:
        PyTree of initialized, sharded JAX arrays with the same structure as param_specs
    """

    # inner function uses out_shardings to ensure arrays are created with
    # the correct sharding
    @partial(jax.jit, out_shardings=shardings)
    def _init_fn(key):
        # determine how many RNG keys are needed (one per tensor !)
        num_leaves = len(
            jax.tree.leaves(param_specs, is_leaf=lambda x: istype(x, ParamSpec))
        )

        # split key into independent keys, one per parameter
        key_iter = iter(random.split(key, num_leaves))

        # initializer each parameter with it's own unique key
        return jax.tree.map(
            lambda info: info.initializer(next(key_iter), info.shape, info.dtype),
            param_specs,
            is_leaf=lambda x: istype(x, ParamSpec),
        )

    return _init_fn(key)


#### Utility Classes ####


@dataclasses.dataclass(frozen=True)
class ParamSpec:
    # meta fields (static => non-trainable)
    shape: Tuple[int, ...] = dataclasses.field(metadata=dict(static=True))
    logical_axes: Axes = dataclasses.field(metadata=dict(static=True))
    initializer: Callable | None = dataclasses.field(
        default=None, metadata=dict(static=True)
    )
    # data fields (trainable)
    dtype: jnp.dtype = dataclasses.field(default=jnp.float32)


class ParamInitializer:
    @classmethod
    def param_specs(cls, *args, **kwargs):
        """
        Define the specifications (ParamSpec) for all parameters in the PyTree.
        """
        raise NotImplementedError

    @classmethod
    def get_sharding(
        cls, mesh: jax.sharding.Mesh, rules, *args, **kwargs
    ) -> jax.sharding.NamedSharding:
        """
        Get a NamedSharding object for the given mesh and rules.

        Args:
            mesh (jax.sharding.Mesh): The mesh of devices to shard across.
            rules: The sharding rules to use.

        Returns:
            jax.sharding.NamedSharding
        """

        param_specs = cls.param_specs(*args, **kwargs)
        return jax.tree.map(
            lambda param_spec: logical_to_sharding(
                param_spec.logical_axes, mesh, rules
            ),
            param_specs,
            is_leaf=lambda x: istype(x, ParamSpec),
        )

    @classmethod
    def init(
        cls, key: jax.random.PRNGKey, mesh: jax.sharding.Mesh, rules, *args, **kwargs
    ):
        """
        Initialize a PyTree of parameter arrays from ParamSpec objects using a given key

        Args:
            key (jax.random.PRNGKey): key for random number generation
            mesh (jax.sharding.Mesh): The mesh of devices to shard across.
            rules: The sharding rules to use.

        Returns:
            PyTree of initialized, sharded JAX arrays
        """

        param_specs = cls.param_specs(*args, **kwargs)
        shardings = cls.get_sharding(mesh, rules, *args, **kwargs)

        leaves, treedef = jax.tree.flatten(
            param_specs, is_leaf=lambda x: istype(x, ParamSpec)
        )
        leaf_shardings = jax.tree.leaves(
            shardings, is_leaf=lambda x: istype(x, NamedSharding)
        )
        return jax.tree.unflatten(
            treedef, _init_leaves(key, tuple(leaves), tuple(leaf_shardings))
        )
