import os

import jax
import numpy as np
import pytest


def pytest_configure():
    """Set up JAX to simulate multiple devices for testing."""
    if "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

    # Limit JAX memory usage for CI environments
    if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"


@pytest.fixture(scope="session")
def jax_devices():
    devices = jax.devices()
    assert len(devices) == 8, f"Expected 8 devices, got {len(devices)}"
    return devices


@pytest.fixture(scope="session")
def mesh(jax_devices):
    return jax.sharding.Mesh(
        devices=np.array(jax_devices).reshape(2, 4), axis_names=("fsdp", "tp")
    )
