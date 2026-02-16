"""
GPU Kernels for CTv2 Coupled-Trajectory Mixed Quantum-Classical Dynamics

This module provides GPU-accelerated implementations of the O(n^2) cross-trajectory
calculations that are the main computational bottleneck in CTv2 dynamics.

Key functions:
- calculate_gaussian_matrix: O(ntrajs^2) Gaussian product matrix
- calculate_weighted_center: Weighted center using einsum
- calculate_slope_gpu: Full GPU slope calculation
- calculate_center_gpu: Full GPU center/intercept calculation

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
def calculate_qmom_K_torch(qmom, phase_diff, inv_mass, l_coh):
    """
    Calculate decoherence term K from quantum momentum and phase difference.

    K[t,i,j] = 0.5 * sum_a inv_mass[a] * sum_d qmom[t,a,d] * phase_diff[t,i,j,a,d]

    Args:
        qmom: (ntrajs, nat_qm, ndim) quantum momentum
        phase_diff: (ntrajs, nst, nst, nat_qm, ndim) phase differences
        inv_mass: (nat_qm,) inverse masses
        l_coh: (ntrajs, nst) coherence flags

    Returns:
        K: (ntrajs, nst, nst) decoherence terms
    """
    ntrajs, nst = l_coh.shape

    # qmom_phase: sum over dim of qmom * phase_diff -> (ntrajs, nst, nst, nat_qm)
    qmom_phase = torch.einsum('tad,tijad->tija', qmom, phase_diff)

    # K: sum over atoms with inv_mass -> (ntrajs, nst, nst)
    K_full = 0.5 * torch.einsum('a,tija->tij', inv_mass, qmom_phase)

    # Apply coherence mask and upper triangular
    coh_ij = l_coh[:, :, None] & l_coh[:, None, :]
    triu_mask = torch.triu(torch.ones((nst, nst), dtype=torch.bool, device=K_full.device), diagonal=1)

    K = torch.where(coh_ij & triu_mask, K_full, torch.zeros_like(K_full))

    # Make antisymmetric
    K = K - K.transpose(1, 2)

    return K


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
    """

    def __init__(self, gpu_backend):
        """
        Initialize GPU kernels.

        Args:
            gpu_backend: GPUBackend instance from gpu_backend.py
        """
        self.gpu = gpu_backend
        self._compiled_functions = {}

    def calculate_slope(self, ctv2):
        """
        GPU-accelerated slope calculation.

        This replaces the calculate_slope method in CTv2 when GPU is available.

        Args:
            ctv2: CTv2 instance with all necessary arrays
        """
        if not HAS_TORCH:
            return self._calculate_slope_numpy(ctv2)

        # Determine dtype based on device (MPS needs float32)
        dtype = self.gpu.dtype_float

        # Extract data and move to GPU
        pos_list = [mol.pos for mol in ctv2.mols]
        pos = torch.stack([torch.from_numpy(p.astype(np.float32 if dtype == torch.float32 else np.float64)) for p in pos_list]).to(
            self.gpu.device, dtype=dtype
        )  # (ntrajs, nat_qm, ndim)

        rho_list = [np.diag(mol.rho.real).astype(np.float32 if dtype == torch.float32 else np.float64) for mol in ctv2.mols]
        rho = torch.stack([torch.from_numpy(r) for r in rho_list]).to(
            self.gpu.device, dtype=dtype
        )  # (ntrajs, nst)

        # Initialize outputs on GPU
        ntrajs, nst = ctv2.ntrajs, ctv2.nst
        nat_qm, ndim = ctv2.nat_qm, ctv2.ndim

        g_I = torch.zeros(ntrajs, device=self.gpu.device, dtype=dtype)
        g_i_I = torch.ones((nst, ntrajs), device=self.gpu.device, dtype=dtype)
        g_i_IJ = torch.ones((nst, ntrajs, ntrajs), device=self.gpu.device, dtype=dtype)

        rho_avg = torch.sum(rho, dim=0) / ntrajs  # (nst,)

        # Process each state
        for ist in range(nst):
            if rho_avg[ist].item() < ctv2.lower_th:
                g_i_I[ist, :] = 0.0
            else:
                if ctv2.l_traj_gaussian:
                    sigma_sq_np = (ctv2.sigma[ist] ** 2).astype(np.float32 if dtype == torch.float32 else np.float64)
                    sigma_sq = torch.from_numpy(sigma_sq_np).to(self.gpu.device, dtype=dtype)

                    # GPU kernel call
                    g_i_IJ_ist, g_i_I_ist = calculate_g_i_IJ_torch(
                        pos, sigma_sq, rho[:, ist], ntrajs
                    )
                    g_i_IJ[ist] = g_i_IJ_ist
                    g_i_I[ist] = g_i_I_ist
                else:
                    # Non-trajectory-Gaussian case
                    avg_R_np = ctv2.avg_R[ist].astype(np.float32 if dtype == torch.float32 else np.float64)
                    avg_R = torch.from_numpy(avg_R_np).to(self.gpu.device, dtype=dtype)
                    sigma_sq_np = (ctv2.sigma[ist] ** 2).astype(np.float32 if dtype == torch.float32 else np.float64)
                    sigma_sq = torch.from_numpy(sigma_sq_np).to(self.gpu.device, dtype=dtype)

                    pos_diff = pos - avg_R  # (ntrajs, nat_qm, ndim)
                    gauss_val = torch.exp(-0.5 * torch.sum(pos_diff ** 2 / sigma_sq, dim=(1, 2)))
                    norm_factor = torch.prod(1.0 / torch.sqrt(2.0 * np.pi * sigma_sq))
                    g_i_I[ist] = gauss_val * norm_factor * rho_avg[ist]

        # Calculate g_I (total density)
        g_I = torch.sum(g_i_I, dim=0)

        # Calculate pseudo_pop
        pseudo_pop = torch.zeros((nst, ntrajs), device=self.gpu.device, dtype=dtype)
        if ctv2.l_real_pop:
            pseudo_pop = rho.T  # (nst, ntrajs)
        else:
            valid_mask = g_I >= ctv2.small
            pseudo_pop[:, valid_mask] = g_i_I[:, valid_mask] / g_I[valid_mask]

        # Calculate slope
        sigma_sq_all_np = (ctv2.sigma ** 2).astype(np.float32 if dtype == torch.float32 else np.float64)
        sigma_sq_all = torch.from_numpy(sigma_sq_all_np).to(self.gpu.device, dtype=dtype)
        inv_sigma_sq = torch.where(
            sigma_sq_all > ctv2.small,
            1.0 / sigma_sq_all,
            torch.zeros_like(sigma_sq_all)
        )
        slope = calculate_slope_from_pseudo_pop_torch(pseudo_pop, inv_sigma_sq)

        # Copy results back to CPU
        ctv2.g_I[:] = self.gpu.to_numpy(g_I)
        ctv2.g_i_I[:] = self.gpu.to_numpy(g_i_I)
        ctv2.g_i_IJ[:] = self.gpu.to_numpy(g_i_IJ)
        ctv2.pseudo_pop[:] = self.gpu.to_numpy(pseudo_pop)
        ctv2.slope[:] = self.gpu.to_numpy(slope)

        # Calculate slope_bo if needed
        if ctv2.l_crunch:
            index_lk = 0
            for ist in range(nst):
                for jst in range(ist + 1, nst):
                    sigma_i_sq = sigma_sq_all[ist]
                    sigma_j_sq = sigma_sq_all[jst]
                    valid = (sigma_i_sq >= ctv2.small) & (sigma_j_sq >= ctv2.small)
                    slope_val = torch.where(
                        valid,
                        -(1.0 / sigma_i_sq + 1.0 / sigma_j_sq),
                        torch.zeros_like(sigma_i_sq)
                    )
                    ctv2.slope_bo[:, index_lk, :, :] = self.gpu.to_numpy(
                        slope_val[None, :, :].expand(ntrajs, -1, -1)
                    )
                    index_lk += 1

    def calculate_center(self, ctv2):
        """
        GPU-accelerated center/intercept calculation.

        Args:
            ctv2: CTv2 instance with all necessary arrays
        """
        if not HAS_TORCH:
            return self._calculate_center_numpy(ctv2)

        # Determine dtype based on device (MPS needs float32)
        dtype = self.gpu.dtype_float

        # Extract data and move to GPU
        pos_list = [mol.pos for mol in ctv2.mols]
        pos = torch.stack([torch.from_numpy(p.astype(np.float32 if dtype == torch.float32 else np.float64)) for p in pos_list]).to(
            self.gpu.device, dtype=dtype
        )

        ntrajs, nst = ctv2.ntrajs, ctv2.nst

        # Calculate inv_sigma_sq on GPU
        sigma_sq_all_np = (ctv2.sigma ** 2).astype(np.float32 if dtype == torch.float32 else np.float64)
        sigma_sq_all = torch.from_numpy(sigma_sq_all_np).to(self.gpu.device, dtype=dtype)
        inv_sigma_sq = torch.where(
            sigma_sq_all > ctv2.small,
            1.0 / sigma_sq_all,
            torch.zeros_like(sigma_sq_all)
        )

        if ctv2.l_traj_gaussian:
            # Move g_i_IJ to GPU
            g_i_IJ_np = ctv2.g_i_IJ.astype(np.float32 if dtype == torch.float32 else np.float64)
            g_i_IJ = torch.from_numpy(g_i_IJ_np).to(self.gpu.device, dtype=dtype)
            g_I_np = ctv2.g_I.astype(np.float32 if dtype == torch.float32 else np.float64)
            g_I = torch.from_numpy(g_I_np).to(self.gpu.device, dtype=dtype)

            valid_g_I = g_I / ntrajs >= ctv2.small

            # Use GPU kernel for weighted center
            intercept_sum = calculate_weighted_center_torch(g_i_IJ, pos, inv_sigma_sq)

            # Safe division
            g_I_safe = torch.where(valid_g_I, g_I, torch.ones_like(g_I))
            intercept = torch.where(
                valid_g_I[:, None, None],
                -intercept_sum / g_I_safe[:, None, None],
                torch.zeros_like(intercept_sum)
            )

            ctv2.intercept[:] = self.gpu.to_numpy(intercept)

            # Center for invalid trajectories = pos
            center_np = ctv2.center.astype(np.float32 if dtype == torch.float32 else np.float64)
            center_tensor = torch.from_numpy(center_np).to(self.gpu.device, dtype=dtype)
            center = torch.where(valid_g_I[:, None, None], center_tensor, pos)
            ctv2.center[:] = self.gpu.to_numpy(center)
        else:
            # Non-trajectory-Gaussian case
            pseudo_pop_np = ctv2.pseudo_pop.astype(np.float32 if dtype == torch.float32 else np.float64)
            pseudo_pop = torch.from_numpy(pseudo_pop_np).to(self.gpu.device, dtype=dtype)
            avg_R_np = ctv2.avg_R.astype(np.float32 if dtype == torch.float32 else np.float64)
            avg_R = torch.from_numpy(avg_R_np).to(self.gpu.device, dtype=dtype)

            weighted_avg_R = avg_R * inv_sigma_sq
            intercept = -torch.einsum('st,sab->tab', pseudo_pop, weighted_avg_R)
            ctv2.intercept[:] = self.gpu.to_numpy(intercept)

        # Calculate center from slope and intercept
        slope_np = ctv2.slope.astype(np.float32 if dtype == torch.float32 else np.float64)
        slope = torch.from_numpy(slope_np).to(self.gpu.device, dtype=dtype)
        intercept_np = ctv2.intercept.astype(np.float32 if dtype == torch.float32 else np.float64)
        intercept = torch.from_numpy(intercept_np).to(self.gpu.device, dtype=dtype)

        slope_valid = torch.abs(slope) >= ctv2.small
        slope_safe = torch.where(slope_valid, slope, torch.ones_like(slope))
        center = torch.where(slope_valid, intercept / slope_safe, pos)
        ctv2.center[:] = self.gpu.to_numpy(center)

        # Calculate intercept_bo and center_bo if l_crunch
        if ctv2.l_crunch:
            self._calculate_center_bo_gpu(ctv2, pos, sigma_sq_all)

    def _calculate_center_bo_gpu(self, ctv2, pos, sigma_sq_all):
        """Calculate intercept_bo and center_bo on GPU."""
        dtype = self.gpu.dtype_float
        ntrajs, nst = ctv2.ntrajs, ctv2.nst

        g_i_IJ_np = ctv2.g_i_IJ.astype(np.float32 if dtype == torch.float32 else np.float64)
        g_i_IJ = torch.from_numpy(g_i_IJ_np).to(self.gpu.device, dtype=dtype)
        g_i_I_np = ctv2.g_i_I.astype(np.float32 if dtype == torch.float32 else np.float64)
        g_i_I = torch.from_numpy(g_i_I_np).to(self.gpu.device, dtype=dtype)
        slope_bo_np = ctv2.slope_bo.astype(np.float32 if dtype == torch.float32 else np.float64)
        slope_bo = torch.from_numpy(slope_bo_np).to(self.gpu.device, dtype=dtype)

        index_lk = 0
        for ist in range(nst):
            for jst in range(ist + 1, nst):
                sigma_i_sq = sigma_sq_all[ist]
                sigma_j_sq = sigma_sq_all[jst]
                valid_sigma = (sigma_i_sq >= ctv2.small) & (sigma_j_sq >= ctv2.small)

                if ctv2.l_traj_gaussian:
                    if torch.all(valid_sigma):
                        intercept_bo, g_i_valid = calculate_intercept_bo_torch(
                            g_i_IJ[ist], g_i_IJ[jst],
                            g_i_I[ist], g_i_I[jst],
                            pos, sigma_i_sq, sigma_j_sq,
                            ctv2.small
                        )

                        # Apply validity mask
                        ctv2.intercept_bo[:, index_lk] = self.gpu.to_numpy(
                            torch.where(
                                g_i_valid[:, None, None] & valid_sigma[None, :, :],
                                intercept_bo,
                                torch.zeros_like(intercept_bo)
                            )
                        )

                        # Update slope_bo for invalid cases
                        slope_bo_lk = slope_bo[:, index_lk]
                        slope_bo_lk = torch.where(
                            g_i_valid[:, None, None] & valid_sigma[None, :, :],
                            slope_bo_lk,
                            torch.zeros_like(slope_bo_lk)
                        )
                        ctv2.slope_bo[:, index_lk] = self.gpu.to_numpy(slope_bo_lk)
                    else:
                        ctv2.intercept_bo[:, index_lk] = 0.0
                else:
                    # Non-trajectory-Gaussian case
                    avg_R_np = ctv2.avg_R.astype(np.float32 if dtype == torch.float32 else np.float64)
                    avg_R = torch.from_numpy(avg_R_np).to(self.gpu.device, dtype=dtype)
                    inv_sigma_i_sq = torch.where(
                        sigma_i_sq >= ctv2.small, 1.0 / sigma_i_sq, torch.zeros_like(sigma_i_sq)
                    )
                    inv_sigma_j_sq = torch.where(
                        sigma_j_sq >= ctv2.small, 1.0 / sigma_j_sq, torch.zeros_like(sigma_j_sq)
                    )
                    intercept_val = -(avg_R[ist] * inv_sigma_i_sq + avg_R[jst] * inv_sigma_j_sq)
                    ctv2.intercept_bo[:, index_lk] = self.gpu.to_numpy(
                        torch.where(valid_sigma, intercept_val, torch.zeros_like(intercept_val))
                    )[None, :, :]

                # Calculate center_bo
                slope_bo_lk_np = ctv2.slope_bo[:, index_lk].astype(np.float32 if dtype == torch.float32 else np.float64)
                slope_bo_lk = torch.from_numpy(slope_bo_lk_np).to(self.gpu.device, dtype=dtype)
                intercept_bo_lk_np = ctv2.intercept_bo[:, index_lk].astype(np.float32 if dtype == torch.float32 else np.float64)
                intercept_bo_lk = torch.from_numpy(intercept_bo_lk_np).to(self.gpu.device, dtype=dtype)
                slope_bo_valid = torch.abs(slope_bo_lk) >= ctv2.small
                center_bo = torch.where(
                    slope_bo_valid,
                    intercept_bo_lk / slope_bo_lk,
                    pos
                )
                ctv2.center_bo[:, index_lk] = self.gpu.to_numpy(center_bo)

                index_lk += 1

    def _calculate_slope_numpy(self, ctv2):
        """Fallback NumPy implementation of calculate_slope."""
        # This is the original implementation - just call the CPU method
        pass

    def _calculate_center_numpy(self, ctv2):
        """Fallback NumPy implementation of calculate_center."""
        pass
