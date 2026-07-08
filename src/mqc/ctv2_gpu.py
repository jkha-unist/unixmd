"""
GPU Kernels for CTv2 Coupled-Trajectory Mixed Quantum-Classical Dynamics

This module provides GPU-accelerated implementations of the O(n^2) cross-trajectory
calculations that are the main computational bottleneck in CTv2 dynamics.

Key functions:
- calculate_gaussian_matrix: O(ntrajs^2) Gaussian product matrix
- calculate_weighted_center: Weighted center using einsum
- calculate_slope_and_center: Fused slope+center calculation with persistent tensors

These functions use PyTorch with @torch.compile for JIT compilation on
Apple Silicon (MPS), NVIDIA GPUs (CUDA), or CPU.
"""

from __future__ import division
import numpy as np

# Try to import PyTorch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _get_compile_decorator():
    """Get appropriate compile decorator based on PyTorch version and device."""
    if not HAS_TORCH:
        return lambda fn: fn

    # Check PyTorch version for torch.compile support (requires 2.0+)
    version = torch.__version__.split('.')
    major = int(version[0])

    # MPS backend doesn't fully support torch.compile yet - it's still experimental
    # and causes errors with symbolic shapes. Disable compilation for now.
    # See: https://github.com/pytorch/pytorch/issues/150121
    if torch.backends.mps.is_available():
        return lambda fn: fn

    if major >= 2:
        # torch.compile available for CUDA
        def conditional_compile(fn):
            try:
                return torch.compile(fn, mode='reduce-overhead')
            except Exception:
                return fn
        return conditional_compile
    else:
        # PyTorch < 2.0, no torch.compile
        return lambda fn: fn


# Get the compile decorator
_compile = _get_compile_decorator()


@_compile
def calculate_gaussian_matrix_torch(pos, sigma_sq):
    """
    Compute O(ntrajs^2) Gaussian product matrix on GPU.

    This computes the Gaussian overlap between all pairs of trajectories:
        G[i,j] = exp(-0.5 * sum_{a,d} (pos[i,a,d] - pos[j,a,d])^2 / sigma_sq[a,d])

    Args:
        pos: (ntrajs, nat_qm, ndim) positions tensor
        sigma_sq: (nat_qm, ndim) squared sigmas tensor

    Returns:
        gauss_val: (ntrajs, ntrajs) Gaussian values
    """
    # pos_diff: (ntrajs, ntrajs, nat_qm, ndim) = pos[i] - pos[j]
    pos_diff = pos[:, None, :, :] - pos[None, :, :, :]

    # Gaussian exponent: sum over atoms and dims of (pos_diff^2 / sigma_sq)
    # sigma_sq is broadcast from (nat_qm, ndim) to (1, 1, nat_qm, ndim)
    gauss_exp = -0.5 * torch.sum(pos_diff ** 2 / sigma_sq, dim=(2, 3))

    return torch.exp(gauss_exp)


@_compile
def calculate_weighted_center_torch(g_i_IJ, pos, inv_sigma_sq):
    """
    Compute weighted center using einsum on GPU.

    This computes the Gaussian-weighted position average for the intercept calculation:
        intercept_sum[i,a,d] = sum_{s,j} g_i_IJ[s,i,j] * pos[j,a,d] * inv_sigma_sq[s,a,d]

    Args:
        g_i_IJ: (nst, ntrajs, ntrajs) Gaussian weights
        pos: (ntrajs, nat_qm, ndim) positions
        inv_sigma_sq: (nst, nat_qm, ndim) inverse squared sigmas

    Returns:
        intercept_sum: (ntrajs, nat_qm, ndim)
    """
    return torch.einsum('sij,jad,sad->iad', g_i_IJ, pos, inv_sigma_sq)


@_compile
def calculate_slope_from_pseudo_pop_torch(pseudo_pop, inv_sigma_sq):
    """
    Calculate slope from pseudo populations and inverse sigma squared.

    slope[t,a,b] = -sum_s pseudo_pop[s,t] * inv_sigma_sq[s,a,b]

    Args:
        pseudo_pop: (nst, ntrajs) pseudo populations
        inv_sigma_sq: (nst, nat_qm, ndim) inverse squared sigmas

    Returns:
        slope: (ntrajs, nat_qm, ndim)
    """
    return -torch.einsum('st,sab->tab', pseudo_pop, inv_sigma_sq)


@_compile
def calculate_g_i_IJ_torch(pos, sigma_sq, rho, ntrajs):
    """
    Calculate full g_i_IJ matrix for a single state.

    g_i_IJ[i,j] = gauss_val[i,j] * norm_factor * rho[j] / ntrajs

    Args:
        pos: (ntrajs, nat_qm, ndim) positions
        sigma_sq: (nat_qm, ndim) squared sigma for this state
        rho: (ntrajs,) diagonal density matrix elements for this state
        ntrajs: number of trajectories

    Returns:
        g_i_IJ: (ntrajs, ntrajs) Gaussian basis matrix
        g_i_I: (ntrajs,) sum over j of g_i_IJ
    """
    # Calculate Gaussian matrix
    gauss_val = calculate_gaussian_matrix_torch(pos, sigma_sq)

    # Normalization factor
    norm_factor = torch.prod(1.0 / torch.sqrt(2.0 * np.pi * sigma_sq))

    # g_i_IJ[i,j] = gauss_val[i,j] * norm_factor * rho[j] / ntrajs
    g_i_IJ = gauss_val * norm_factor * rho[None, :] / ntrajs

    # g_i_I[i] = sum_j g_i_IJ[i,j]
    g_i_I = torch.sum(g_i_IJ, dim=1)

    return g_i_IJ, g_i_I


@_compile
def calculate_g_i_IJ_fully_batched_torch(pos, sigma_sq_all, rho, ntrajs):
    """
    Fully batched g_i_IJ calculation - all states computed in parallel.

    This computes the Gaussian matrix for all states simultaneously by
    broadcasting sigma_sq across states.

    Args:
        pos: (ntrajs, nat_qm, ndim) positions
        sigma_sq_all: (nst, nat_qm, ndim) squared sigma for all states
        rho: (ntrajs, nst) diagonal density matrix elements
        ntrajs: number of trajectories

    Returns:
        g_i_IJ: (nst, ntrajs, ntrajs) Gaussian basis matrix for all states
        g_i_I: (nst, ntrajs) sum over j of g_i_IJ for all states
    """
    nst = sigma_sq_all.shape[0]

    # pos_diff: (ntrajs, ntrajs, nat_qm, ndim)
    pos_diff = pos[:, None, :, :] - pos[None, :, :, :]
    pos_diff_sq = pos_diff ** 2

    # Broadcast sigma_sq to (nst, 1, 1, nat_qm, ndim) for batched division
    # pos_diff_sq: (ntrajs, ntrajs, nat_qm, ndim) -> (1, ntrajs, ntrajs, nat_qm, ndim)
    # Result: (nst, ntrajs, ntrajs, nat_qm, ndim)
    scaled_diff_sq = pos_diff_sq[None, :, :, :, :] / sigma_sq_all[:, None, None, :, :]

    # Sum over (nat_qm, ndim) to get (nst, ntrajs, ntrajs)
    gauss_exp = -0.5 * torch.sum(scaled_diff_sq, dim=(3, 4))
    gauss_val = torch.exp(gauss_exp)  # (nst, ntrajs, ntrajs)

    # Normalization: (nst,)
    # torch.prod doesn't support tuple dim, so flatten and prod
    norm_vals = 1.0 / torch.sqrt(2.0 * np.pi * sigma_sq_all)  # (nst, nat_qm, ndim)
    norm_factors = torch.prod(norm_vals.view(nst, -1), dim=1)  # (nst,)

    # g_i_IJ[s, i, j] = gauss_val[s,i,j] * norm[s] * rho[j,s] / ntrajs
    # rho: (ntrajs, nst) -> (nst, 1, ntrajs)
    g_i_IJ = gauss_val * norm_factors[:, None, None] * rho.T[:, None, :] / ntrajs

    # g_i_I[s, i] = sum_j g_i_IJ[s, i, j]
    g_i_I = torch.sum(g_i_IJ, dim=2)

    return g_i_IJ, g_i_I


@_compile
def calculate_intercept_bo_torch(g_i_IJ_ist, g_i_IJ_jst, g_i_I_ist, g_i_I_jst,
                                  pos, sigma_i_sq, sigma_j_sq, small):
    """
    Calculate intercept_bo for a state pair with trajectory-centered Gaussians.

    Args:
        g_i_IJ_ist: (ntrajs, ntrajs) Gaussian basis for state ist
        g_i_IJ_jst: (ntrajs, ntrajs) Gaussian basis for state jst
        g_i_I_ist: (ntrajs,) normalization for state ist
        g_i_I_jst: (ntrajs,) normalization for state jst
        pos: (ntrajs, nat_qm, ndim) positions
        sigma_i_sq: (nat_qm, ndim) squared sigma for state ist
        sigma_j_sq: (nat_qm, ndim) squared sigma for state jst
        small: threshold for valid values

    Returns:
        intercept_bo: (ntrajs, nat_qm, ndim)
        g_i_valid: (ntrajs,) validity mask
    """
    ntrajs = pos.shape[0]

    # Validity check
    g_i_valid = (g_i_I_ist / ntrajs >= small) & (g_i_I_jst / ntrajs >= small)

    # Safe division factors
    g_i_I_ist_safe = torch.where(g_i_valid, g_i_I_ist, torch.ones_like(g_i_I_ist))
    g_i_I_jst_safe = torch.where(g_i_valid, g_i_I_jst, torch.ones_like(g_i_I_jst))

    # Term 1: sum_j (g_i_IJ[ist,i,j] * pos[j,a,d] / sigma_i_sq[a,d]) / g_i_I[ist,i]
    term1_sum = torch.einsum('ij,jad->iad', g_i_IJ_ist, pos) / sigma_i_sq[None, :, :]
    term1 = term1_sum / g_i_I_ist_safe[:, None, None]

    # Term 2: sum_j (g_i_IJ[jst,i,j] * pos[j,a,d] / sigma_j_sq[a,d]) / g_i_I[jst,i]
    term2_sum = torch.einsum('ij,jad->iad', g_i_IJ_jst, pos) / sigma_j_sq[None, :, :]
    term2 = term2_sum / g_i_I_jst_safe[:, None, None]

    intercept_val = -(term1 + term2)

    # Apply validity mask
    intercept_bo = torch.where(g_i_valid[:, None, None], intercept_val,
                               torch.zeros_like(intercept_val))

    return intercept_bo, g_i_valid


class CTv2GPUKernels:
    """
    GPU kernel manager for CTv2 calculations.

    This class manages the GPU execution of cross-trajectory calculations,
    handling data transfer and kernel dispatch.

    Features:
    - Persistent GPU tensors to avoid repeated CPU->GPU transfers
    - Fused slope+center calculation to avoid g_i_IJ round-trip
    - Batched state computation to reduce kernel launch overhead
    """

    def __init__(self, gpu_backend):
        """
        Initialize GPU kernels.

        Args:
            gpu_backend: GPUBackend instance from gpu_backend.py
        """
        self.gpu = gpu_backend
        self._compiled_functions = {}

        # Persistent GPU tensors (initialized on first use)
        self._persistent_tensors = None
        self._tensor_shapes = None

    def _init_persistent_tensors(self, ctv2):
        """
        Initialize persistent GPU tensors for a CTv2 instance.

        These tensors remain on GPU across MD steps, avoiding repeated transfers.
        """
        if not HAS_TORCH:
            return

        torch = self.gpu._torch
        dtype = self.gpu.dtype_float
        device = self.gpu.device

        ntrajs = ctv2.ntrajs
        nst = ctv2.nst
        nat_qm = ctv2.nat_qm
        ndim = ctv2.ndim
        nst_pair = nst * (nst - 1) // 2

        # Check if tensors need reinitialization (shape changed)
        new_shapes = (ntrajs, nst, nat_qm, ndim, nst_pair)
        if self._tensor_shapes == new_shapes and self._persistent_tensors is not None:
            return  # Already initialized with correct shapes

        self._tensor_shapes = new_shapes
        self._persistent_tensors = {
            # Input tensors (synced from CPU each step)
            'pos': torch.zeros((ntrajs, nat_qm, ndim), device=device, dtype=dtype),
            'rho': torch.zeros((ntrajs, nst), device=device, dtype=dtype),
            'sigma_sq': torch.zeros((nst, nat_qm, ndim), device=device, dtype=dtype),
            'avg_R': torch.zeros((nst, nat_qm, ndim), device=device, dtype=dtype),

            # Intermediate tensors (computed on GPU, reused)
            'g_I': torch.zeros(ntrajs, device=device, dtype=dtype),
            'g_i_I': torch.zeros((nst, ntrajs), device=device, dtype=dtype),
            'g_i_IJ': torch.zeros((nst, ntrajs, ntrajs), device=device, dtype=dtype),
            'pseudo_pop': torch.zeros((nst, ntrajs), device=device, dtype=dtype),
            'inv_sigma_sq': torch.zeros((nst, nat_qm, ndim), device=device, dtype=dtype),

            # Output tensors
            'slope': torch.zeros((ntrajs, nat_qm, ndim), device=device, dtype=dtype),
            'intercept': torch.zeros((ntrajs, nat_qm, ndim), device=device, dtype=dtype),
            'center': torch.zeros((ntrajs, nat_qm, ndim), device=device, dtype=dtype),
            'slope_bo': torch.zeros((ntrajs, nst_pair, nat_qm, ndim), device=device, dtype=dtype),
            'intercept_bo': torch.zeros((ntrajs, nst_pair, nat_qm, ndim), device=device, dtype=dtype),
            'center_bo': torch.zeros((ntrajs, nst_pair, nat_qm, ndim), device=device, dtype=dtype),
        }

    def sync_to_gpu(self, ctv2):
        """
        Sync molecular data from CPU to persistent GPU tensors.

        This method efficiently updates only the data that changes each step:
        - pos: nuclear positions (changes every step)
        - rho: density matrix (changes every step)
        - sigma_sq: Gaussian widths (computed on CPU in calculate_sigma)
        - avg_R: average positions (computed on CPU in calculate_sigma)

        Args:
            ctv2: CTv2 instance with molecular data
        """
        if not HAS_TORCH or self._persistent_tensors is None:
            return

        torch = self.gpu._torch
        dtype = self.gpu.dtype_float
        tensors = self._persistent_tensors

        # Sync pos: stack and transfer (this is the main cost)
        pos_np = np.stack([mol.pos for mol in ctv2.mols])
        if dtype == torch.float32:
            pos_np = pos_np.astype(np.float32)
        tensors['pos'].copy_(torch.from_numpy(pos_np))

        # Sync rho diagonal
        rho_np = np.stack([np.diag(mol.rho.real) for mol in ctv2.mols])
        if dtype == torch.float32:
            rho_np = rho_np.astype(np.float32)
        tensors['rho'].copy_(torch.from_numpy(rho_np))

        # Sync sigma_sq (computed fresh each step in calculate_sigma)
        sigma_sq_np = ctv2.sigma ** 2
        if dtype == torch.float32:
            sigma_sq_np = sigma_sq_np.astype(np.float32)
        tensors['sigma_sq'].copy_(torch.from_numpy(sigma_sq_np))

        # Sync avg_R
        avg_R_np = ctv2.avg_R.copy()
        if dtype == torch.float32:
            avg_R_np = avg_R_np.astype(np.float32)
        tensors['avg_R'].copy_(torch.from_numpy(avg_R_np))

        # Precompute inv_sigma_sq
        sigma_sq = tensors['sigma_sq']
        tensors['inv_sigma_sq'] = torch.where(
            sigma_sq > ctv2.small,
            1.0 / sigma_sq,
            torch.zeros_like(sigma_sq)
        )

    def calculate_slope_and_center(self, ctv2):
        """
        Fused GPU calculation with persistent tensors AND batched state computation.

        This is the optimized version that:
        1. Uses persistent GPU tensors to avoid repeated allocation
        2. Keeps g_i_IJ on GPU throughout (no round-trip transfer)
        3. Computes g_i_IJ for ALL states in a single batched operation
        4. Minimizes kernel launch overhead

        Args:
            ctv2: CTv2 instance with all necessary arrays
        """
        if not HAS_TORCH:
            return False

        torch = self.gpu._torch
        dtype = self.gpu.dtype_float
        ntrajs, nst = ctv2.ntrajs, ctv2.nst
        nat_qm, ndim = ctv2.nat_qm, ctv2.ndim

        # Initialize persistent tensors if needed
        self._init_persistent_tensors(ctv2)

        # Sync data from CPU to GPU
        self.sync_to_gpu(ctv2)

        # Get references to persistent tensors
        t = self._persistent_tensors
        pos = t['pos']
        rho = t['rho']
        sigma_sq_all = t['sigma_sq']
        inv_sigma_sq = t['inv_sigma_sq']
        g_I = t['g_I']
        g_i_I = t['g_i_I']
        g_i_IJ = t['g_i_IJ']
        pseudo_pop = t['pseudo_pop']
        slope = t['slope']
        intercept = t['intercept']
        center = t['center']
        slope_bo = t['slope_bo']
        intercept_bo = t['intercept_bo']
        center_bo = t['center_bo']

        # Reset intermediate tensors
        g_I.zero_()
        g_i_I.zero_()
        g_i_IJ.zero_()
        pseudo_pop.zero_()

        rho_avg = torch.sum(rho, dim=0) / ntrajs  # (nst,)

        # ===== BATCHED: Calculate g_i_IJ for all states at once =====
        if ctv2.l_traj_gaussian:
            # Identify valid states (rho_avg >= lower_th)
            valid_states = rho_avg >= ctv2.lower_th

            if torch.any(valid_states):
                # Use fully batched kernel for all valid states
                g_i_IJ_all, g_i_I_all = calculate_g_i_IJ_fully_batched_torch(
                    pos, sigma_sq_all, rho, ntrajs
                )

                # Apply validity mask
                valid_mask = valid_states[:, None]  # (nst, 1)
                g_i_IJ.copy_(torch.where(valid_mask[:, :, None], g_i_IJ_all, torch.zeros_like(g_i_IJ_all)))
                g_i_I.copy_(torch.where(valid_mask, g_i_I_all, torch.zeros_like(g_i_I_all)))
        else:
            # Non-trajectory Gaussian mode - process per state
            for ist in range(nst):
                if rho_avg[ist].item() >= ctv2.lower_th:
                    avg_R = t['avg_R'][ist]
                    sigma_sq = sigma_sq_all[ist]
                    pos_diff = pos - avg_R
                    gauss_val = torch.exp(-0.5 * torch.sum(pos_diff ** 2 / sigma_sq, dim=(1, 2)))
                    norm_factor = torch.prod(1.0 / torch.sqrt(2.0 * np.pi * sigma_sq))
                    g_i_I[ist] = gauss_val * norm_factor * rho_avg[ist]

        # ===== Calculate g_I and pseudo_pop =====
        g_I.copy_(torch.sum(g_i_I, dim=0))

        if ctv2.l_real_pop:
            pseudo_pop.copy_(rho.T)
        else:
            valid_mask = g_I >= ctv2.small
            pseudo_pop[:, valid_mask] = g_i_I[:, valid_mask] / g_I[valid_mask]

        # ===== Calculate slope =====
        slope.copy_(calculate_slope_from_pseudo_pop_torch(pseudo_pop, inv_sigma_sq))

        # ===== Calculate intercept =====
        if ctv2.l_traj_gaussian:
            valid_g_I = g_I / ntrajs >= ctv2.small
            intercept_sum = calculate_weighted_center_torch(g_i_IJ, pos, inv_sigma_sq)
            g_I_safe = torch.where(valid_g_I, g_I, torch.ones_like(g_I))
            intercept.copy_(torch.where(
                valid_g_I[:, None, None],
                -intercept_sum / g_I_safe[:, None, None],
                torch.zeros_like(intercept_sum)
            ))
        else:
            avg_R = t['avg_R']
            weighted_avg_R = avg_R * inv_sigma_sq
            intercept.copy_(-torch.einsum('st,sab->tab', pseudo_pop, weighted_avg_R))

        # ===== Calculate center =====
        slope_valid = torch.abs(slope) >= ctv2.small
        slope_safe = torch.where(slope_valid, slope, torch.ones_like(slope))
        center.copy_(torch.where(slope_valid, intercept / slope_safe, pos))

        # ===== Calculate slope_bo, intercept_bo, center_bo if l_crunch =====
        if ctv2.l_crunch:
            slope_bo.zero_()
            intercept_bo.zero_()

            index_lk = 0
            for ist in range(nst):
                for jst in range(ist + 1, nst):
                    sigma_i_sq = sigma_sq_all[ist]
                    sigma_j_sq = sigma_sq_all[jst]
                    valid_sigma = (sigma_i_sq >= ctv2.small) & (sigma_j_sq >= ctv2.small)

                    slope_val = torch.where(
                        valid_sigma,
                        -(1.0 / sigma_i_sq + 1.0 / sigma_j_sq),
                        torch.zeros_like(sigma_i_sq)
                    )
                    slope_bo[:, index_lk, :, :] = slope_val[None, :, :]

                    if ctv2.l_traj_gaussian:
                        g_i_valid = ((g_i_I[ist] / ntrajs >= ctv2.small) &
                                     (g_i_I[jst] / ntrajs >= ctv2.small))
                        all_sigma_valid = torch.all(valid_sigma)

                        if all_sigma_valid:
                            intercept_bo_val, _ = calculate_intercept_bo_torch(
                                g_i_IJ[ist], g_i_IJ[jst],
                                g_i_I[ist], g_i_I[jst],
                                pos, sigma_i_sq, sigma_j_sq,
                                ctv2.small
                            )
                            intercept_bo[:, index_lk] = torch.where(
                                g_i_valid[:, None, None] & valid_sigma[None, :, :],
                                intercept_bo_val,
                                torch.zeros_like(intercept_bo_val)
                            )
                            slope_bo[:, index_lk] = torch.where(
                                g_i_valid[:, None, None] & valid_sigma[None, :, :],
                                slope_bo[:, index_lk],
                                torch.zeros_like(slope_bo[:, index_lk])
                            )
                    else:
                        inv_sigma_i_sq = torch.where(
                            sigma_i_sq >= ctv2.small, 1.0 / sigma_i_sq, torch.zeros_like(sigma_i_sq)
                        )
                        inv_sigma_j_sq = torch.where(
                            sigma_j_sq >= ctv2.small, 1.0 / sigma_j_sq, torch.zeros_like(sigma_j_sq)
                        )
                        avg_R = t['avg_R']
                        intercept_val = -(avg_R[ist] * inv_sigma_i_sq + avg_R[jst] * inv_sigma_j_sq)
                        intercept_bo[:, index_lk] = torch.where(
                            valid_sigma[None, :, :], intercept_val[None, :, :], torch.zeros_like(intercept_val[None, :, :])
                        )

                    slope_bo_valid = torch.abs(slope_bo[:, index_lk]) >= ctv2.small
                    center_bo[:, index_lk] = torch.where(
                        slope_bo_valid,
                        intercept_bo[:, index_lk] / slope_bo[:, index_lk],
                        pos
                    )
                    index_lk += 1

        # ===== Transfer results back to CPU =====
        ctv2.g_I[:] = self.gpu.to_numpy(g_I)
        ctv2.g_i_I[:] = self.gpu.to_numpy(g_i_I)
        ctv2.g_i_IJ[:] = self.gpu.to_numpy(g_i_IJ)
        ctv2.pseudo_pop[:] = self.gpu.to_numpy(pseudo_pop)
        ctv2.slope[:] = self.gpu.to_numpy(slope)
        ctv2.intercept[:] = self.gpu.to_numpy(intercept)
        ctv2.center[:] = self.gpu.to_numpy(center)

        if ctv2.l_crunch:
            ctv2.slope_bo[:] = self.gpu.to_numpy(slope_bo)
            ctv2.intercept_bo[:] = self.gpu.to_numpy(intercept_bo)
            ctv2.center_bo[:] = self.gpu.to_numpy(center_bo)

        if ctv2.l_traj_gaussian:
            valid_g_I_np = self.gpu.to_numpy(g_I / ntrajs >= ctv2.small)
            for itraj in range(ntrajs):
                if not valid_g_I_np[itraj]:
                    ctv2.center[itraj] = ctv2.mols[itraj].pos

        return True
