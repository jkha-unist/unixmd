"""
GPU Kernels Package for PyUNIxMD

This package contains GPU-accelerated implementations of computationally
intensive operations used in mixed quantum-classical dynamics.

Modules:
    rk4_torch: Batched RK4 electronic propagation using PyTorch
"""

from __future__ import division

# Check for PyTorch availability
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from .rk4_torch import rk4_step_batch, rk4_propagate_batch
