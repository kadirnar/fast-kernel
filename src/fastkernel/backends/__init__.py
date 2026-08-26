"""Kernel backends: capability probes, helpers and starter templates.

A probe never *forbids* anything: it records what compiles on this machine today (evidence) so the
agent can pick a backend, and what failed with the exact error so it can fix the environment
(e.g. point CUDA_HOME at the pip-installed nvcc) instead of declaring a hardware limitation.
"""
from .base import BACKENDS, device_capabilities, probe_all  # noqa: F401
