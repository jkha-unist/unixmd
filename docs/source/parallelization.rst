.. _CPU Parallelization:

===========================
CPU Parallelization
===========================

CTv2 supports CPU parallelization for distributing QM calculations across multiple processor cores.
This is the primary bottleneck in coupled-trajectory dynamics, as each trajectory requires an
independent electronic structure calculation at every time step.


Usage
---------------------------

Enable parallelization by setting the ``ncpus`` parameter in the CTv2 constructor:

.. code-block:: python

   md = mqc.CTv2(molecules=mols, istates=istates, ..., ncpus=4)

- ``ncpus=1`` (default): Serial execution
- ``ncpus=N``: Use N worker processes for QM calculations

The pool size is ``min(ncpus, ntrajs)`` — there is no benefit to having more workers than trajectories.


Three-Phase Loop Architecture
-------------------------------

At each MD time step, the dynamics loop is organized into phases that alternate
between serial and parallel execution:

.. code-block:: text

   for istep in range(nsteps):
       ┌─────────────────────────────────────────┐
       │ Pre-QM phase (serial)                    │
       │   - Nuclear half-step position update    │
       │   - Backup BO data                       │
       └─────────────────────────────────────────┘
                         │
                         ▼
       ┌─────────────────────────────────────────┐
       │ QM phase (parallel via pool.map)         │
       │   Worker 0: mol[0] → qm.get_data()      │
       │   Worker 1: mol[1] → qm.get_data()      │
       │   Worker 2: mol[2] → qm.get_data()      │
       │   ...                                    │
       └─────────────────────────────────────────┘
                         │
                         ▼
       ┌─────────────────────────────────────────┐
       │ Post-QM phase (serial)                   │
       │   - Velocity update                      │
       │   - Electronic propagation               │
       │   - Coherence/decoherence checks         │
       └─────────────────────────────────────────┘
                         │
                         ▼
       ┌─────────────────────────────────────────┐
       │ Cross-trajectory phase (serial or GPU)   │
       │   - calculate_sigma()                    │
       │   - calculate_slope() / _center()        │
       │   - calculate_qmom()                     │
       │   - set_avg_pop_cons()                   │
       └─────────────────────────────────────────┘

Only the QM phase is parallelized. The cross-trajectory phase uses vectorized NumPy
(or GPU kernels when ``use_gpu`` is enabled).


Worker Functions
---------------------------

The parallelization uses Python's ``multiprocessing`` module with three key functions
defined at module level in ``src/mqc/ctv2.py``:

``_qm_get_data_worker(args)``
  The main worker function. Each invocation:

  1. Looks up (or creates) a persistent deep copy of the QM calculator for the given trajectory index
  2. Calls ``mol.reset_bo()`` to clear BO data
  3. Calls ``qm.get_data()`` to run the electronic structure calculation
  4. Calls ``mol.adjust_nac()`` if needed for NAC phase alignment
  5. Returns the updated Molecule object

``_init_qm_worker(qm, nthreads)``
  Pool initializer called once per worker process. Stores the QM template for on-demand
  deep copying and limits BLAS threads via ``_limit_blas_threads()``.

``_limit_blas_threads(n)``
  Limits BLAS thread count via ctypes calls to the math library's C-level thread-setting
  functions (``openblas_set_num_threads`` or ``MKL_Set_Num_Threads``).


BLAS Thread Management
---------------------------

After ``fork()``, the BLAS thread pool is already initialized, so environment variables
(``OMP_NUM_THREADS``, ``MKL_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``) have no effect.
PyUNIxMD calls the library's C-level function directly via ``ctypes``:

.. code-block:: python

   import ctypes, ctypes.util
   for lib_name, setter in [('openblas', 'openblas_set_num_threads'),
                             ('mkl_rt', 'MKL_Set_Num_Threads')]:
       path = ctypes.util.find_library(lib_name)
       if path:
           lib = ctypes.CDLL(path)
           getattr(lib, setter)(nthreads)

The number of BLAS threads per worker is computed as:

.. code-block:: python

   nthreads_per_worker = max(1, os.cpu_count() // pool_size)

This prevents thread oversubscription where N workers x M BLAS threads > total CPU cores.


Pool Configuration
---------------------------

- **Fork context**: Uses ``multiprocessing.get_context('fork')`` for zero-copy memory sharing
- **Pool size**: ``min(ncpus, ntrajs)``
- **Persistent QM objects**: Each worker maintains a dictionary of deep-copied QM calculators
  keyed by trajectory index, ensuring stateful attributes (e.g., ``scr_qm_dir``) remain
  consistent across MD steps
- **Picklability check**: Before creating the pool, PyUNIxMD tests that the QM and Molecule
  objects can be pickled. If not, it falls back to serial execution with a warning.


Limitations
---------------------------

- The QM calculator object must be picklable (serializable). Some QM interfaces with
  complex C extensions or file handles may not support this.
- Thermostat is not supported with parallelization (raises an error).
- MM (molecular mechanics) calculations are not parallelized.
- The ``fork`` context is not available on Windows. Use Linux or macOS for parallelization.


Combining GPU and CPU Parallelization
----------------------------------------

GPU acceleration and CPU parallelization target different phases of the dynamics loop
and can be used together:

.. code-block:: python

   md = mqc.CTv2(molecules=mols, istates=istates, ...,
                  ncpus=4, use_gpu=True)

- ``ncpus > 1``: Parallelizes the QM calculations (per-trajectory, embarrassingly parallel)
- ``use_gpu=True``: Accelerates the cross-trajectory slope/center/quantum momentum calculations

For large trajectory counts (ntrajs > 100) with expensive QM calculations, combining both
provides the best performance.
