import contextlib
import os

import jax
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec


def pytest_configure():
    """Set up JAX to simulate multiple devices for testing."""
    if "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

    # Limit JAX memory usage for CI environments
    if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"


# Valid 2D mesh shapes for an 8-device TPU node.
# (fsdp_size, tp_size)
#   (1, 8) -> pure tensor parallelism
#   (2, 4) -> balanced (default)
_MESH_SHAPES = [(1, 8), (2, 4)]


@pytest.fixture(scope="session")
def jax_devices():
    devices = jax.devices()
    assert len(devices) == 8, f"Expected 8 devices, got {len(devices)}"
    return devices


@pytest.fixture(scope="session")
def mesh(jax_devices):
    return jax.sharding.Mesh(
        devices=np.array(jax_devices).reshape(2, 4),
        axis_names=("fsdp", "tp"),
    )


# ---------------------------------------------------------------------------
# Shared test helpers  (ideally lives in conftest.py, not individual modules)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def maybe_mesh(mesh):
    """Context manager that enters the mesh when given one, else no-ops."""
    if mesh is not None:
        with mesh:
            yield
    else:
        yield


def shard_batch(batch, mesh):
    """Shard a batch dict across the mesh's 'fsdp' axis.

    Returns the batch unchanged when ``mesh`` is ``None``.
    """
    if mesh is None:
        return batch
    fsdp_sharding = NamedSharding(mesh, PartitionSpec("fsdp", None))
    replicate_sharding = NamedSharding(mesh, PartitionSpec())
    sharded = {}
    for k, v in batch.items():
        if isinstance(v, jax.Array) and v.ndim >= 1:
            sharded[k] = jax.device_put(v, fsdp_sharding)
        elif isinstance(v, jax.Array):
            sharded[k] = jax.device_put(v, replicate_sharding)
        else:
            sharded[k] = v
    return sharded


def assert_on_mesh_devices(arr, mesh):
    """Assert that an array is placed on devices belonging to the mesh.

    No-op when ``mesh`` is ``None``.
    """
    if mesh is None:
        return
    if hasattr(arr, "sharding"):
        device_set = arr.sharding.device_set
        assert device_set.issubset(set(mesh.devices.flat)), (
            f"Array sharded on {device_set} but mesh only has {set(mesh.devices.flat)}"
        )
