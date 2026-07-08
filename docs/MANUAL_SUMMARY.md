# PyUNIxMD Manual Summary

This document provides a standalone reference for the CTv2, SHXFv2, GPU acceleration, CPU parallelization, vectorization patterns, and code structure of PyUNIxMD. For the full Sphinx documentation, build with `cd docs && make html`.

---

## 1. Code Structure

### Directory Layout

```
PyUNIxMD/
├── src/                         # Main source code
│   ├── molecule.py              # Molecule & State classes
│   ├── polariton.py             # QED polariton support
│   ├── thermostat.py            # Thermostat implementations
│   ├── misc.py                  # Atomic data, unit conversions, FileManager
│   │
│   ├── mqc/                     # MQC dynamics methods
│   │   ├── mqc.py               # Base MQC class (Template Method pattern)
│   │   ├── bomd.py              # Born-Oppenheimer MD
│   │   ├── eh.py                # Ehrenfest dynamics
│   │   ├── sh.py                # Fewest-switches surface hopping (FSSH)
│   │   ├── shxf.py              # SHXF (SH + exact factorization)
│   │   ├── ehxf.py              # EhXF (Ehrenfest + exact factorization)
│   │   ├── ct.py                # Coupled-trajectory MQC (CTMQC)
│   │   ├── ctv2.py              # Coupled-trajectory v2 (CTv2)
│   │   ├── shxfv2.py            # DISH-XF (SHXFv2)
│   │   ├── gpu_backend.py       # GPU backend abstraction (PyTorch/NumPy)
│   │   └── ctv2_gpu.py          # GPU kernels for CTv2
│   │
│   ├── mqc_qed/                 # QED-coupled MQC variants
│   ├── qm/                      # Quantum chemistry interfaces
│   │   ├── qm_calculator.py     # Base QM calculator
│   │   ├── columbus/, molpro/, gaussian09/, qchem/, turbomole/, terachem/, dftbplus/, gamess/
│   │   └── model/               # Built-in models (SAC, DAC, DAG, ECR, Shin-Metiu, File_IO)
│   ├── mm/                      # Molecular mechanics (Tinker)
│   ├── qed/                     # QED model Hamiltonians (Jaynes-Cummings)
│   ├── cpa/                     # Classical path approximation
│   └── lib/                     # Cython/C electronic propagators
│       ├── mqc/                 # el_propagator*.pyx/.c
│       ├── mqc_qed/             # QED propagators
│       └── cioverlap/           # CI overlap calculations
│
├── lib/                         # Compiled shared libraries (.so)
├── docs/                        # Sphinx documentation
├── tests/                       # Test suite (pytest)
├── examples/                    # Example input scripts
└── setup.py                     # Build configuration
```

### MQC Class Hierarchy

```
MQC (base — Template Method pattern)
├── Single-trajectory
│   ├── BOMD       — Born-Oppenheimer MD
│   ├── Eh         — Ehrenfest dynamics
│   ├── SH         — Fewest-switches surface hopping
│   ├── SHXF       — SH + exact factorization decoherence
│   ├── EhXF       — Ehrenfest + exact factorization decoherence
│   └── SHXFv2     — DISH-XF (decoherence-induced SH with XF)
└── Multi-trajectory
    ├── CT         — Coupled-trajectory MQC (CTMQC)
    └── CTv2       — Coupled-trajectory v2 (vectorized, GPU/parallel)
```

### Cython Build Pipeline

```bash
python3 setup.py build_ext -b ./lib/
```

Produces: `libmqc`, `libmqcxf`, `libmqcxfv2`, `libctmqc`, `libctmqcv2`, `libcioverlap`, `libmqc_qed`, `libmqcxf_qed`

---

## 2. CTv2 Method

### Overview

CTv2 is an enhanced coupled-trajectory MQC method featuring:
- **CRUNCH** projected quantum momentum (state-pair-resolved decoherence)
- **State-wise momentum** for phase calculations
- **Population conservation** schemes (none, scaling, shift)
- **GPU acceleration** and **CPU parallelization**
- Vectorized NumPy/einsum operations

### Constructor Parameters

```python
mqc.CTv2(
    molecules,                    # List of Molecule objects (required)
    istates=None,                 # List of initial states (required)
    thermostat=None,              # Not supported yet
    dt=0.5,                       # Time interval
    nsteps=1000,                  # Total nuclear steps
    nesteps=20,                   # Electronic sub-steps
    elec_object="coefficient",    # Only 'coefficient' supported
    propagator="rk4",             # Only 'rk4' supported
    l_print_dm=True,              # Print density matrix files
    l_adj_nac=True,               # Adjust NAC phases
    rho_threshold=0.01,           # Coherence threshold
    init_coefs=None,              # Initial coefficients (2D list)
    l_crunch=True,                # CRUNCH projected quantum momentum
    l_dc_w_mom=True,              # State-wise momentum for phase
    l_traj_gaussian=False,        # Trajectory-centered Gaussians
    t_cons=2,                     # Pop conservation: 0=none, 1=scaling, 2=shift
    l_etot0=True,                 # Constant E_tot at t=0
    l_lap=False,                  # Laplacian ENC term
    l_real_pop=True,              # Use |C_j|^2 for populations
    t_pc=1,                       # Phase correction: 0/1/2
    l_en_cons=False,              # Enforce energy conservation
    artifact_expon=0.2,           # Width exponent (traj Gaussians only)
    l_asymp=False,                # Terminate in asymptotic region
    x_fin=25.0,                   # Asymptotic boundary (a.u.)
    use_gpu=False,                # GPU: False/True/'auto'
    ncpus=1,                      # CPU parallelization
    unit_dt="fs",                 # Time unit: 'fs' or 'au'
    out_freq=1,                   # Output frequency
    verbosity=0                   # 0/1/2
)
```

### Simulation Loop

```
for istep in range(nsteps):
    # 1. Pre-QM: half-step nuclear position update (serial, per trajectory)
    # 2. QM: electronic structure calculations (parallel if ncpus > 1)
    # 3. Post-QM: velocity update, electronic propagation, coherence checks (serial)
    # 4. Cross-trajectory: sigma, slope/center, qmom, K, pop conservation (vectorized/GPU)
    # 5. Output: write md files for each trajectory
```

### Key Algorithms

- **Quantum momentum**: Gradient of log nuclear density, computed from Gaussian fitting
- **CRUNCH**: State-pair-resolved quantum momentum G_{nu,ij} instead of total P_nu
- **Population conservation**: Shift scheme adds correction to maintain avg population sum = 1
- **State-wise momentum**: Computed from BO energy and total energy conservation

### Output Files (per trajectory in `TRAJ_N/md/`)

| File | Description |
|------|-------------|
| `MOVIE.xyz` | Trajectory with energies |
| `DENSITY` | Populations and coherences |
| `NACME` | Nonadiabatic couplings |
| `PHASE_i` | Phase term (verbosity >= 2) |
| `MOM_i` | State-wise momentum (verbosity >= 2) |
| `K_i_j` | Decoherence K matrix (verbosity >= 2) |
| `SIGMA_i` | Gaussian width (verbosity >= 2) |
| `PSEUDOPOP` | Pseudo populations (verbosity >= 2) |

### Example

```python
from molecule import Molecule
import qm, mqc
from misc import data
import numpy as np

ntraj = 200
data["X1"] = 2000.  # mass in a.u.

np.random.seed(1234)
pos_list = np.random.normal(loc=-25., scale=2.0, size=ntraj)
mom_list = np.random.normal(loc=30., scale=0.25, size=ntraj)

mols, istates = [], []
for itraj in range(ntraj):
    geom = f"\n    1\n    comment\n    X1   {pos_list[itraj]}  {mom_list[itraj]/2000.}\n    "
    mol = Molecule(geometry=geom, ndim=1, nstates=2, ndof=1, unit_pos='au', l_model=True)
    mols.append(mol)
    istates.append(0)

bo = qm.model.DAG(molecule=mols[0])
md = mqc.CTv2(molecules=mols, istates=istates, dt=1., nsteps=1000, nesteps=20,
              elec_object="coefficient", l_adj_nac=False, rho_threshold=0.01,
              unit_dt="au", out_freq=10, l_crunch=True, l_dc_w_mom=True,
              l_traj_gaussian=False, t_cons=2, l_etot0=True)
md.run(qm=bo, output_dir="./")
```

---

## 3. SHXFv2 Method (DISH-XF)

### Overview

SHXFv2 (Decoherence-Induced Surface Hopping with Exact Factorization) extends SHXF with:
- **CRUNCH** decoherence (state-pair-resolved quantum momentum)
- **Phase correction** schemes with fine-grained control
- Modified defaults: `aux_econs_viol='fix'`
- Option to disable hopping (`l_no_hop=True`)

### Constructor Parameters

```python
mqc.SHXFv2(
    molecule,                     # Molecule object (required)
    istate=0,                     # Initial state
    thermostat=None,
    dt=0.5, nsteps=1000, nesteps=20,
    elec_object="coefficient",
    propagator="rk4",
    l_print_dm=True, l_adj_nac=True,
    hop_rescale="augment",        # 'energy'/'velocity'/'momentum'/'augment'
    hop_reject="reverse",         # 'keep'/'reverse'
    rho_threshold=0.01,
    sigma=None,                   # Required: float or list of floats
    init_coef=None,
    l_econs_state=True,
    aux_econs_viol="fix",         # 'fix'/'collapse'
    l_crunch=True,                # CRUNCH decoherence
    t_pc=0,                       # Phase correction: 0/1/2
    l_exact_pc=True,              # Exact phase correction
    l_pc_w_phase_term=False,
    l_divide_rho=False,
    l_pc_nac=False,
    l_pc_rescale=False,
    l_no_hop=False,               # Disable hopping
    l_asymp=False, x_fin=25.0,
    unit_dt="fs", out_freq=1, verbosity=0
)
```

### Auxiliary Trajectories

Each BO state has an auxiliary trajectory. The velocity is scaled by alpha:

```
vel_aux[k] = alpha_k * vel_true
alpha_k = sqrt((E_tot_k - E_k) / KE_true)
```

When `E_tot_k - E_k < 0`, the trajectory violates energy conservation:
- `'fix'`: Freeze the auxiliary trajectory
- `'collapse'`: Destroy it and renormalize coefficients

### Hopping Logic

1. **hop_prob**: Fewest-switches formula + force hop (deterministic hop on full decoherence)
2. **hop_check**: Compare accumulated probability against random number
3. **evaluate_hop**: Rescale velocity (quadratic solver for momentum/augment rescaling)

### Decoherence Lifecycle

1. **check_coherence**: State population > threshold → enter coherence
2. **check_decoherence**: Population < threshold while coherent → decohere, reset auxiliary
3. **set_decoherence**: Collapse coefficient, renormalize

### Output Files

| File | Description |
|------|-------------|
| `DENSITY` | Populations and coherences |
| `NACME` | Nonadiabatic couplings |
| `SHSTATE` | Running state |
| `SHPROB` | Hopping probabilities |
| `QMOM_i_j` | Quantum momentum (verbosity >= 2) |
| `AUX_PHASE_i` | Phase term (verbosity >= 2) |
| `AUX_MOVIE_i.xyz` | Auxiliary trajectory (verbosity >= 2) |

---

## 4. GPU Acceleration

### Setup

```bash
pip install torch
```

### Activation

```python
# Constructor parameter
md = mqc.CTv2(..., use_gpu=True)    # Force GPU
md = mqc.CTv2(..., use_gpu='auto')  # Auto-detect

# Environment variable (overrides constructor)
export PYUNIXMD_USE_GPU=true
export PYUNIXMD_USE_GPU=auto
```

### Device Priority

1. Apple Silicon (MPS) — float32 precision
2. NVIDIA CUDA — float64 precision
3. CPU fallback with PyTorch

### Architecture

**GPUBackend** (`gpu_backend.py`):
- Unified interface for PyTorch and NumPy
- `to_device()`, `to_numpy()`, `einsum()`, `synchronize()`

**CTv2GPUKernels** (`ctv2_gpu.py`):
- Persistent GPU tensors (allocated once, reused across steps)
- Fused pipeline: sync → compute g_i_IJ → slope/center → transfer back
- Batched state computation

### GPU Kernel Functions

| Function | Description |
|----------|-------------|
| `calculate_gaussian_matrix_torch` | O(ntrajs^2) Gaussian product matrix |
| `calculate_weighted_center_torch` | Weighted center via einsum |
| `calculate_slope_from_pseudo_pop_torch` | Slope from pseudo populations |
| `calculate_g_i_IJ_fully_batched_torch` | All-state batched Gaussian basis |
| `calculate_intercept_bo_torch` | State-pair intercept (traj Gaussians) |

### torch.compile

- CUDA: `@torch.compile(mode='reduce-overhead')` for JIT (PyTorch 2.0+)
- MPS: Disabled (incomplete backend support)

### When GPU Helps

- ntrajs > ~100 with trajectory-centered Gaussians: significant speedup
- ntrajs < ~50: CPU is typically faster due to transfer overhead

---

## 5. Vectorization Patterns

NumPy vectorization is used throughout PyUNIxMD to eliminate Python-level loops over atoms, states, and trajectories.

Index conventions: `t`=trajectory, `s`=state, `i,j`=state pair, `a`=atom, `b,d`=dimension

### Core Operations (Molecule, Polariton)

| Pattern | Subscript | Methods |
|---------|-----------|---------|
| Kinetic energy | `'i,ij->'` | `Molecule.update_kinetic()`, `Polariton.update_kinetic()` |
| NACME from NAC vectors | `np.tensordot(nac, vel, axes=([2,3],[0,1]))` | `Molecule.get_nacme()`, `Polariton.get_nacme()` |
| NAC phase adjustment | `np.sum(nac**2, axis=(2,3))` + sign broadcasting | `Molecule.adjust_nac()`, `Polariton.adjust_nac()` |

### Ehrenfest Force (Eh, CT, CTv2, QED variants)

| Pattern | Subscript | Description |
|---------|-----------|-------------|
| Adiabatic force | `'i,i...->...'` | Population-weighted sum of state forces |
| Non-adiabatic force | `'k,k,k...->...'` | Energy-diff x coupling x coherence |

These patterns appear in `Eh`, `CT`, `CTv2`, `mqc_qed.Eh`, and `mqc_qed.CT`.

### Velocity Rescaling Quadratic (SH, SHXF, SHXFv2)

| Pattern | Subscript | Description |
|---------|-----------|-------------|
| NAC norm (velocity) | `'i,ij->'` | Mass-weighted NAC squared norm |
| NAC-vel dot (velocity) | `'i,ij,ij->'` | Mass-weighted NAC-velocity product |
| NAC norm (momentum) | `'i,ij->'` (with 1/mass) | Inverse-mass-weighted NAC norm |
| NAC-vel dot (momentum) | `'ij,ij->'` | NAC-velocity Frobenius product |

Identical patterns in `SH.evaluate_hop()`, `SHXF.evaluate_hop()`, `SHXFv2.evaluate_hop()`.

### XF Force (EhXF)

| Pattern | Subscript | Description |
|---------|-----------|-------------|
| XF decoherence force | `'ij,ijab->ab'` | Weight x phase_diff contraction |

### Coupled-Trajectory (CT)

| Pattern | Subscript | Description |
|---------|-----------|-------------|
| CT force | `'ij,ijab->ab'` | Weight x phase_diff for decoherence force |
| K matrix | `'i,tpi->tp'` | Mass-weighted quantum momentum kinetic energy |
| Population smoothing | `'ij,jk->ik'` | Gaussian-weighted population matrix product |
| Weighted center | `'ijad,jad->iad'` | Trajectory-weighted position average |

### CTv2 Cross-Trajectory

| Pattern | Subscript | Description |
|---------|-----------|-------------|
| CT force | `'ij,ijkl,ij->kl'` | K x phase_diff x rho contraction |
| K matrix (step 1) | `'tad,tijad->tija'` | Qmom x phase_diff over dimensions |
| K matrix (step 2) | `'a,tija->tij'` | Mass-weighted sum over atoms |
| Slope | `'st,sab->tab'` | Population x inverse sigma |
| Intercept (traj gauss) | `'sij,jad,sad->iad'` | 3-operand Gaussian-weighted position |
| Intercept (single gauss) | `'st,sab->tab'` | Population x weighted average position |
| Laplacian | `'a,ija,ija->ij'` | Mass-weighted Laplacian contraction |
| Sigma (tensordot) | `tensordot(pos, rho, axes=(0,0))` | Trajectory-weighted statistics |

### QED (Jaynes-Cummings)

| Pattern | Subscript | Description |
|---------|-----------|-------------|
| Polarization coupling | `'k,ijk->ij'` | Field polarization x transition dipole |

### Model Systems (Shin-Metiu)

Energy difference matrix and NAC via broadcasting:
```python
energy_diff = ws[np.newaxis, :] - ws[:, np.newaxis]  # (nst, nst)
nac_full = dVijs / np.where(energy_diff != 0, energy_diff, 1.)
```

### Broadcasting Conventions

**State-pair outer products** (CT, CTv2):
```python
rho_ij = rho_diag[:, np.newaxis] * rho_diag[np.newaxis, :]  # (nst, nst)
```

**Phase differences** — 5D tensor from 4D (CTv2):
```python
phase_diff = phase[:, :, np.newaxis, :, :] - phase[:, np.newaxis, :, :, :]
```

**Coherence mask** — 3D boolean from 2D (CTv2):
```python
coh_ij = l_coh[:, :, np.newaxis] & l_coh[:, np.newaxis, :]
```

**NAC sign broadcasting** — 2D to 4D (Molecule):
```python
self.nac *= sign_matrix[:, :, np.newaxis, np.newaxis]
```

**Auxiliary trajectory init** — 2D to 3D (SHXF, SHXFv2, EhXF):
```python
self.aux.pos[:, :, :] = mol_pos[np.newaxis, :, :]
```

**Batch data extraction** (CT, CTv2):
```python
pos = np.array([mol.pos for mol in self.mols])   # (ntrajs, nat, ndim)
rho = np.array([np.diag(mol.rho.real) for mol in self.mols])  # (ntrajs, nst)
```

### Matrix Symmetry Operations

- **Antisymmetry** (NACME, K): `np.triu(M, k=1) - np.triu(M, k=1).T`
- **3D antisymmetry** (CTv2 K): `self.K -= self.K.transpose(0, 2, 1)`
- **Hermiticity** (SH rho): `np.triu(rho) + np.triu(rho, k=1).conj().T`

### Safe Division with np.where

Used throughout to avoid division by zero in NAC, quantum momentum, and Gaussian calculations:
```python
inv_sigma_sq = np.where(sigma_sq > small, 1.0 / sigma_sq, 0.0)
center = np.where(slope_valid, intercept / slope_safe, pos)
```

---

## 6. CPU Parallelization

### Usage

```python
md = mqc.CTv2(molecules=mols, istates=istates, ..., ncpus=4)
```

### Three-Phase Loop

```
for istep:
    Pre-QM (serial):   Nuclear half-step, backup BO
    QM (parallel):     pool.map(_qm_get_data_worker, ...) — one worker per trajectory
    Post-QM (serial):  Velocity update, electronic propagation, coherence checks
    Cross-traj (serial/GPU): sigma, slope, center, qmom, K, pop conservation
```

### Worker Functions

- `_qm_get_data_worker(args)`: Runs `reset_bo + qm.get_data + adjust_nac` for one trajectory. Each trajectory index gets a persistent deep copy of the QM calculator.
- `_init_qm_worker(qm, nthreads)`: Pool initializer; stores QM template and limits BLAS threads.
- `_limit_blas_threads(n)`: Calls `openblas_set_num_threads` or `MKL_Set_Num_Threads` via ctypes (env vars don't work post-fork).

### BLAS Thread Management

```python
nthreads_per_worker = max(1, os.cpu_count() // pool_size)
```

Prevents oversubscription: N workers x M BLAS threads <= total CPU cores.

### Pool Configuration

- Context: `multiprocessing.get_context('fork')`
- Pool size: `min(ncpus, ntrajs)`
- Picklability check before pool creation (falls back to serial with warning)
- Persistent QM objects per worker (keyed by trajectory index)

### Combining GPU + CPU

```python
md = mqc.CTv2(..., ncpus=4, use_gpu=True)
```

- `ncpus > 1`: Parallelizes QM calculations (per-trajectory)
- `use_gpu=True`: Accelerates cross-trajectory calculations (slope/center/qmom)

### Limitations

- QM object must be picklable
- Thermostat not supported with parallelization
- `fork` context not available on Windows
