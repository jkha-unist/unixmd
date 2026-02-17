"""
GPU Backend Abstraction for PyUNIxMD

Provides a unified interface for PyTorch (with MPS/CUDA support) and NumPy backends.
This allows seamless switching between GPU and CPU execution.

Usage:
    from mqc.gpu_backend import get_backend

    gpu = get_backend(use_gpu=False)  # CPU mode (default)

    # Move array to GPU
    tensor = gpu.to_device(numpy_array)

    # Move back to NumPy
    result = gpu.to_numpy(tensor)

Environment variable override:
    export PYUNIXMD_USE_GPU=false  # Force CPU mode
    export PYUNIXMD_USE_GPU=true   # Force GPU mode (error if unavailable)
    export PYUNIXMD_USE_GPU=auto   # Auto-detect (default)
"""

from __future__ import division
import os
import numpy as np


class GPUBackend:
    """Backend abstraction for GPU/CPU tensor operations.

    Attributes:
        backend (str): 'torch' if using PyTorch, 'numpy' for NumPy fallback
        device: torch.device object if using PyTorch, None otherwise
        dtype_float: Default floating point dtype
        dtype_complex: Default complex dtype
    """

    def __init__(self, use_gpu=False):
        """Initialize GPU backend.

        Args:
            use_gpu: False/'false' (CPU, default), True/'true' (force GPU),
                     'auto' (detect GPU)
        """
        self.backend = 'numpy'
        self.device = None
        self.dtype_float = np.float64
        self.dtype_complex = np.complex128
        self._torch = None

        # Normalize use_gpu parameter
        if isinstance(use_gpu, str):
            use_gpu = use_gpu.lower()
        elif isinstance(use_gpu, bool):
            use_gpu = 'true' if use_gpu else 'false'

        # Check environment variable override
        env_use_gpu = os.environ.get('PYUNIXMD_USE_GPU', '').lower()
        if env_use_gpu in ('true', 'false', 'auto'):
            use_gpu = env_use_gpu

        if use_gpu == 'false':
            return

        try:
            import torch
            self._torch = torch

            # Determine device
            device = None
            if torch.backends.mps.is_available():
                # Apple Silicon (M1/M2/M3)
                device = torch.device('mps')
            elif torch.cuda.is_available():
                # NVIDIA GPU
                device = torch.device('cuda')
            elif use_gpu == 'true':
                # Force GPU requested but none available, use CPU with PyTorch
                device = torch.device('cpu')

            if device is not None or use_gpu == 'true':
                self.device = device if device is not None else torch.device('cpu')
                self.backend = 'torch'
                # MPS doesn't support float64, use float32 for MPS
                if self.device is not None and self.device.type == 'mps':
                    self.dtype_float = torch.float32
                    self.dtype_complex = torch.complex64
                else:
                    self.dtype_float = torch.float64
                    self.dtype_complex = torch.complex128

        except ImportError:
            if use_gpu == 'true':
                raise ImportError(
                    "PyTorch is required for GPU acceleration. "
                    "Install with: pip install torch"
                )

    @property
    def is_gpu(self):
        """Check if using GPU (MPS or CUDA)."""
        if self.backend != 'torch' or self.device is None:
            return False
        return self.device.type in ('mps', 'cuda')

    @property
    def device_name(self):
        """Get human-readable device name."""
        if self.backend == 'numpy':
            return 'CPU (NumPy)'
        if self.device is None:
            return 'CPU (PyTorch)'
        if self.device.type == 'mps':
            return 'Apple Silicon GPU (MPS)'
        if self.device.type == 'cuda':
            device_name = self._torch.cuda.get_device_name(self.device)
            return f'NVIDIA GPU ({device_name})'
        return f'CPU (PyTorch)'

    def to_device(self, arr, dtype=None):
        """Move NumPy array to device tensor.

        Args:
            arr: NumPy array or existing tensor
            dtype: Optional dtype override

        Returns:
            Tensor on device if using PyTorch, original array if NumPy backend
        """
        if self.backend == 'numpy':
            if dtype is not None:
                return arr.astype(dtype)
            return arr

        # Handle already-tensor case
        if self._torch.is_tensor(arr):
            if arr.device == self.device:
                return arr
            return arr.to(self.device)

        # Convert NumPy to tensor
        tensor = self._torch.from_numpy(np.ascontiguousarray(arr))
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor.to(self.device)

    def to_numpy(self, tensor):
        """Convert tensor back to NumPy array.

        Args:
            tensor: PyTorch tensor or NumPy array

        Returns:
            NumPy array
        """
        if self.backend == 'numpy':
            return tensor

        if not self._torch.is_tensor(tensor):
            return tensor

        # MPS tensors need to be moved to CPU first
        if tensor.device.type == 'mps':
            return tensor.cpu().numpy()
        elif tensor.device.type == 'cuda':
            return tensor.cpu().numpy()
        else:
            return tensor.numpy()

    def zeros(self, shape, dtype=None, complex_dtype=False):
        """Create zero-filled array/tensor.

        Args:
            shape: Tuple of dimensions
            dtype: Optional dtype override
            complex_dtype: If True, use complex dtype

        Returns:
            Zero tensor on device or NumPy array
        """
        if dtype is None:
            dtype = self.dtype_complex if complex_dtype else self.dtype_float

        if self.backend == 'numpy':
            np_dtype = np.complex128 if complex_dtype else np.float64
            return np.zeros(shape, dtype=np_dtype)

        return self._torch.zeros(shape, dtype=dtype, device=self.device)

    def ones(self, shape, dtype=None, complex_dtype=False):
        """Create one-filled array/tensor.

        Args:
            shape: Tuple of dimensions
            dtype: Optional dtype override
            complex_dtype: If True, use complex dtype

        Returns:
            One tensor on device or NumPy array
        """
        if dtype is None:
            dtype = self.dtype_complex if complex_dtype else self.dtype_float

        if self.backend == 'numpy':
            np_dtype = np.complex128 if complex_dtype else np.float64
            return np.ones(shape, dtype=np_dtype)

        return self._torch.ones(shape, dtype=dtype, device=self.device)

    def stack(self, arrays, dim=0):
        """Stack arrays along a new dimension.

        Args:
            arrays: List of arrays/tensors
            dim: Dimension to stack along

        Returns:
            Stacked tensor or array
        """
        if self.backend == 'numpy':
            return np.stack(arrays, axis=dim)

        # Convert to tensors if needed
        tensors = [self.to_device(arr) for arr in arrays]
        return self._torch.stack(tensors, dim=dim)

    def einsum(self, subscripts, *operands):
        """Perform einsum operation.

        Args:
            subscripts: Einsum subscript string
            *operands: Input arrays/tensors

        Returns:
            Result tensor or array
        """
        if self.backend == 'numpy':
            return np.einsum(subscripts, *operands)

        tensors = [self.to_device(op) for op in operands]
        return self._torch.einsum(subscripts, *tensors)

    def exp(self, x):
        """Element-wise exponential."""
        if self.backend == 'numpy':
            return np.exp(x)
        return self._torch.exp(x)

    def sqrt(self, x):
        """Element-wise square root."""
        if self.backend == 'numpy':
            return np.sqrt(x)
        return self._torch.sqrt(x)

    def sum(self, x, dim=None, keepdim=False):
        """Sum over dimensions."""
        if self.backend == 'numpy':
            return np.sum(x, axis=dim, keepdims=keepdim)
        return self._torch.sum(x, dim=dim, keepdim=keepdim)

    def prod(self, x, dim=None):
        """Product over dimensions."""
        if self.backend == 'numpy':
            return np.prod(x, axis=dim)
        return self._torch.prod(x, dim=dim)

    def abs(self, x):
        """Absolute value."""
        if self.backend == 'numpy':
            return np.abs(x)
        return self._torch.abs(x)

    def where(self, condition, x, y):
        """Element-wise where operation."""
        if self.backend == 'numpy':
            return np.where(condition, x, y)
        return self._torch.where(condition, x, y)

    def synchronize(self):
        """Synchronize GPU operations (wait for completion)."""
        if self.backend == 'torch' and self.device is not None:
            if self.device.type == 'cuda':
                self._torch.cuda.synchronize()
            elif self.device.type == 'mps':
                self._torch.mps.synchronize()


# Global backend instance (lazy initialization)
_backend = None


def get_backend(use_gpu=False):
    """Get or create the global GPU backend instance.

    Args:
        use_gpu: False/'false' (CPU, default), True/'true' (force GPU),
                 'auto' (detect GPU)

    Returns:
        GPUBackend instance
    """
    global _backend
    if _backend is None:
        _backend = GPUBackend(use_gpu)
    return _backend


def reset_backend():
    """Reset the global backend (useful for testing)."""
    global _backend
    _backend = None
