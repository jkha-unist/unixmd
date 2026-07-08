.. _GPU Acceleration:

===========================
GPU Acceleration
===========================

CTv2 supports GPU acceleration for the cross-trajectory calculations that dominate
the computational cost of coupled-trajectory dynamics. GPU offloading is optional and
requires PyTorch.


User Guide
---------------------------

Requirements
''''''''''''''''''''''''''''

- PyTorch (``pip install torch``)
- Supported devices: Apple Silicon (MPS), NVIDIA GPU (CUDA), or CPU fallback

Activation
''''''''''''''''''''''''''''

There are three ways to enable GPU acceleration:

1. **Constructor parameter** (recommended):

   .. code-block:: python

      md = mqc.CTv2(molecules=mols, istates=istates, ..., use_gpu=True)

      # Or auto-detect
      md = mqc.CTv2(molecules=mols, istates=istates, ..., use_gpu='auto')

2. **Environment variable** (overrides constructor):

   .. code-block:: bash

      export PYUNIXMD_USE_GPU=true    # Force GPU
      export PYUNIXMD_USE_GPU=auto    # Auto-detect
      export PYUNIXMD_USE_GPU=false   # Force CPU

3. **Auto mode**: Detects available GPU hardware automatically.

Device Priority
''''''''''''''''''''''''''''

When GPU is enabled, PyUNIxMD selects a device in this order:

1. **Apple Silicon (MPS)** — M1/M2/M3/M4 chips via ``torch.device('mps')``
2. **NVIDIA CUDA** — via ``torch.device('cuda')``
3. **CPU fallback** — if ``use_gpu=True`` but no GPU hardware, PyTorch runs on CPU

.. note:: MPS (Apple Silicon) uses float32 precision instead of float64 due to hardware limitations.
   This has minimal impact on dynamics accuracy for typical simulations.

When Does GPU Help?
''''''''''''''''''''''''''''

The GPU-accelerated kernels target the O(ntrajs\ :sup:`2`) cross-trajectory Gaussian calculations.

- **Many trajectories (ntrajs > ~100)**: GPU provides significant speedup for the slope, center,
  and quantum momentum calculations.
- **Few trajectories (ntrajs < ~50)**: CPU is typically faster due to GPU transfer overhead.
- **Trajectory-centered Gaussians** (``l_traj_gaussian=True``): Largest GPU benefit since the
  full O(ntrajs\ :sup:`2`) Gaussian matrix is computed.


Developer Guide
---------------------------

GPUBackend Class
''''''''''''''''''''''''''''

The ``GPUBackend`` class (``src/mqc/gpu_backend.py``) provides a unified interface that
transparently switches between PyTorch and NumPy:

.. code-block:: python

   from mqc.gpu_backend import get_backend

   gpu = get_backend(use_gpu='auto')

   # Move array to device
   tensor = gpu.to_device(numpy_array)

   # Perform einsum (dispatches to torch.einsum or np.einsum)
   result = gpu.einsum('ij,jk->ik', a, b)

   # Move result back to NumPy
   numpy_result = gpu.to_numpy(result)

   # Synchronize GPU (no-op on CPU)
   gpu.synchronize()

Key attributes and methods:

+------------------------+------------------------------------------------------+
| Attribute/Method       | Description                                          |
+========================+======================================================+
| ``backend``            | ``'torch'`` or ``'numpy'``                           |
+------------------------+------------------------------------------------------+
| ``device``             | ``torch.device`` object or *None*                    |
+------------------------+------------------------------------------------------+
| ``is_gpu``             | *True* if device is MPS or CUDA                      |
+------------------------+------------------------------------------------------+
| ``device_name``        | Human-readable device name string                    |
+------------------------+------------------------------------------------------+
| ``to_device(arr)``     | Move NumPy array to device tensor                    |
+------------------------+------------------------------------------------------+
| ``to_numpy(tensor)``   | Convert tensor back to NumPy array                   |
+------------------------+------------------------------------------------------+
| ``einsum(sub, ...)``   | Perform einsum (dispatches to backend)               |
+------------------------+------------------------------------------------------+
| ``synchronize()``      | Wait for GPU operations to complete                  |
+------------------------+------------------------------------------------------+


CTv2GPUKernels Class
''''''''''''''''''''''''''''

The ``CTv2GPUKernels`` class (``src/mqc/ctv2_gpu.py``) manages GPU execution with:

- **Persistent GPU tensors**: Allocated once and reused across MD steps to avoid repeated allocation
- **Fused pipeline**: Keeps intermediate results (``g_i_IJ``) on GPU throughout the calculation
- **Batched state computation**: Computes Gaussian matrix for all states in a single kernel

The main entry point is ``calculate_slope_and_center(ctv2)``, which executes the full pipeline:

1. ``sync_to_gpu(ctv2)`` — Transfer positions, density, sigma from CPU to persistent GPU tensors
2. Compute ``g_i_IJ`` for all states (batched or per-state)
3. Compute ``g_I``, ``pseudo_pop``, ``slope``, ``intercept``, ``center``
4. If ``l_crunch``: compute ``slope_bo``, ``intercept_bo``, ``center_bo``
5. Transfer results back to CPU

GPU Kernel Functions
''''''''''''''''''''''''''''

+---------------------------------------------+-------------------------------------------------+
| Function                                    | Description                                     |
+=============================================+=================================================+
| ``calculate_gaussian_matrix_torch``         | O(ntrajs\ :sup:`2`) Gaussian product matrix     |
+---------------------------------------------+-------------------------------------------------+
| ``calculate_weighted_center_torch``         | Weighted center via einsum                       |
+---------------------------------------------+-------------------------------------------------+
| ``calculate_slope_from_pseudo_pop_torch``   | Slope from pseudo populations                   |
+---------------------------------------------+-------------------------------------------------+
| ``calculate_g_i_IJ_fully_batched_torch``    | All-state batched Gaussian basis matrix          |
+---------------------------------------------+-------------------------------------------------+
| ``calculate_intercept_bo_torch``            | State-pair intercept for trajectory Gaussians    |
+---------------------------------------------+-------------------------------------------------+

torch.compile Behavior
''''''''''''''''''''''''''''

- **CUDA**: ``@torch.compile(mode='reduce-overhead')`` is applied for JIT compilation (requires PyTorch 2.0+)
- **MPS**: ``torch.compile`` is disabled due to incomplete MPS backend support for symbolic shapes
- **CPU**: Falls back to eager mode if ``torch.compile`` is unavailable
