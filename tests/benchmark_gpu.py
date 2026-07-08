#!/usr/bin/env python3
"""
GPU Benchmark for PyUNIxMD CTv2 Method

This script benchmarks the GPU-accelerated cross-trajectory calculations
in the CTv2 coupled-trajectory mixed quantum-classical dynamics method.

Usage:
    python benchmark_gpu.py [--ntrajs N1 N2 ...] [--iterations N] [--warmup N]

Examples:
    python benchmark_gpu.py --ntrajs 100 500 1000 2000
    python benchmark_gpu.py --ntrajs 500 --iterations 100

Requirements:
    - PyTorch (pip install torch)
    - NumPy
"""

from __future__ import division
import sys
import os
import time
import argparse
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mqc.gpu_backend import get_backend, reset_backend


def create_mock_ctv2_data(ntrajs, nat_qm=10, ndim=3, nst=3):
    """Create mock CTv2 data for benchmarking."""

    class MockMol:
        def __init__(self, nat_qm, ndim, nst):
            self.pos = np.random.randn(nat_qm, ndim)
            self.vel = np.random.randn(nat_qm, ndim)
            self.mass = np.ones(nat_qm) * 2000.0  # typical atomic mass in a.u.
            self.rho = np.eye(nst, dtype=complex) / nst
            self.nacme = np.random.randn(nst, nst) * 0.01
            self.nacme = (self.nacme - self.nacme.T) / 2  # antisymmetric

        @property
        def states(self):
            return [MockState(i, self.rho[i, i].real) for i in range(self.rho.shape[0])]

    class MockState:
        def __init__(self, idx, pop):
            self.energy = -0.5 + idx * 0.1
            self.coef = np.sqrt(pop) + 0j

    class MockCTv2:
        def __init__(self, ntrajs, nat_qm, ndim, nst):
            self.ntrajs = ntrajs
            self.nat_qm = nat_qm
            self.ndim = ndim
            self.nst = nst
            self.nst_pair = nst * (nst - 1) // 2

            self.mols = [MockMol(nat_qm, ndim, nst) for _ in range(ntrajs)]

            # Initialize arrays
            self.g_I = np.zeros(ntrajs)
            self.g_i_I = np.ones((nst, ntrajs))
            self.g_i_IJ = np.ones((nst, ntrajs, ntrajs))
            self.sigma = np.ones((nst, nat_qm, ndim)) * 0.5
            self.avg_R = np.zeros((nst, nat_qm, ndim))
            self.pseudo_pop = np.zeros((nst, ntrajs))
            self.slope = np.zeros((ntrajs, nat_qm, ndim))
            self.intercept = np.zeros((ntrajs, nat_qm, ndim))
            self.center = np.zeros((ntrajs, nat_qm, ndim))
            self.slope_bo = np.zeros((ntrajs, self.nst_pair, nat_qm, ndim))
            self.intercept_bo = np.zeros((ntrajs, self.nst_pair, nat_qm, ndim))
            self.center_bo = np.zeros((ntrajs, self.nst_pair, nat_qm, ndim))

            # Thresholds and flags
            self.small = 1e-8
            self.lower_th = 0.01
            self.l_traj_gaussian = True
            self.l_crunch = True
            self.l_real_pop = True

    return MockCTv2(ntrajs, nat_qm, ndim, nst)


def benchmark_gaussian_matrix(ntrajs_list, iterations=10, warmup=3):
    """Benchmark Gaussian matrix calculation."""
    print("\n" + "=" * 70)
    print("Gaussian Matrix Calculation Benchmark (O(ntrajs^2))")
    print("=" * 70)

    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False
        print("PyTorch not available, skipping GPU benchmark")
        return

    results = []

    for ntrajs in ntrajs_list:
        print(f"\nntrajs = {ntrajs}")
        print("-" * 50)

        nat_qm, ndim = 10, 3

        # Create test data
        pos_np = np.random.randn(ntrajs, nat_qm, ndim)
        sigma_sq = np.ones((nat_qm, ndim)) * 0.25

        # NumPy baseline
        def numpy_gaussian():
            pos_diff = pos_np[:, None, :, :] - pos_np[None, :, :, :]
            gauss_exp = -0.5 * np.sum(pos_diff ** 2 / sigma_sq, axis=(2, 3))
            return np.exp(gauss_exp)

        # Warmup
        for _ in range(warmup):
            numpy_gaussian()

        # Benchmark NumPy
        times_numpy = []
        for _ in range(iterations):
            start = time.perf_counter()
            result_numpy = numpy_gaussian()
            times_numpy.append(time.perf_counter() - start)

        mean_numpy = np.mean(times_numpy) * 1000
        std_numpy = np.std(times_numpy) * 1000
        print(f"  NumPy:  {mean_numpy:.3f} +/- {std_numpy:.3f} ms")

        # PyTorch GPU
        if has_torch:
            from mqc.ctv2_gpu import calculate_gaussian_matrix_torch

            # Detect device
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                device_name = "MPS (Apple Silicon)"
            elif torch.cuda.is_available():
                device = torch.device('cuda')
                device_name = f"CUDA ({torch.cuda.get_device_name()})"
            else:
                device = torch.device('cpu')
                device_name = "CPU (PyTorch)"

            # MPS needs float32
            if device.type == 'mps':
                pos_torch = torch.from_numpy(pos_np.astype(np.float32)).to(device)
                sigma_sq_torch = torch.from_numpy(sigma_sq.astype(np.float32)).to(device)
            else:
                pos_torch = torch.from_numpy(pos_np).to(device)
                sigma_sq_torch = torch.from_numpy(sigma_sq).to(device)

            def torch_gaussian():
                result = calculate_gaussian_matrix_torch(pos_torch, sigma_sq_torch)
                if device.type in ('mps', 'cuda'):
                    torch.mps.synchronize() if device.type == 'mps' else torch.cuda.synchronize()
                return result

            # Warmup
            for _ in range(warmup):
                torch_gaussian()

            # Benchmark
            times_torch = []
            for _ in range(iterations):
                start = time.perf_counter()
                result_torch = torch_gaussian()
                times_torch.append(time.perf_counter() - start)

            mean_torch = np.mean(times_torch) * 1000
            std_torch = np.std(times_torch) * 1000
            speedup = mean_numpy / mean_torch

            print(f"  {device_name}:  {mean_torch:.3f} +/- {std_torch:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")

            # Verify correctness
            if device.type == 'mps':
                result_check = result_torch.cpu().numpy()
            else:
                result_check = result_torch.numpy() if device.type == 'cpu' else result_torch.cpu().numpy()

            max_diff = np.max(np.abs(result_numpy - result_check))
            print(f"  Max difference: {max_diff:.2e}")

            results.append({
                'ntrajs': ntrajs,
                'numpy_ms': mean_numpy,
                'torch_ms': mean_torch,
                'speedup': speedup,
                'device': device_name
            })

    return results


def benchmark_full_slope(ntrajs_list, iterations=5, warmup=2):
    """Benchmark full slope calculation."""
    print("\n" + "=" * 70)
    print("Full Slope Calculation Benchmark")
    print("=" * 70)

    try:
        import torch
        from mqc.ctv2_gpu import CTv2GPUKernels
        has_torch = True
    except ImportError:
        has_torch = False
        print("PyTorch not available, skipping GPU benchmark")
        return

    results = []

    for ntrajs in ntrajs_list:
        print(f"\nntrajs = {ntrajs}")
        print("-" * 50)

        # Create mock data
        ctv2 = create_mock_ctv2_data(ntrajs)

        # NumPy baseline (simulate the calculation)
        def numpy_slope():
            pos = np.array([mol.pos for mol in ctv2.mols])
            rho = np.array([np.diag(mol.rho.real) for mol in ctv2.mols])

            for ist in range(ctv2.nst):
                pos_diff = pos[:, None, :, :] - pos[None, :, :, :]
                sigma_sq = ctv2.sigma[ist] ** 2
                gauss_exp = -0.5 * np.sum(pos_diff ** 2 / sigma_sq, axis=(2, 3))
                gauss_val = np.exp(gauss_exp)
                norm_factor = np.prod(1.0 / np.sqrt(2.0 * np.pi * sigma_sq))
                ctv2.g_i_IJ[ist] = gauss_val * norm_factor * rho[:, ist][None, :] / ntrajs
                ctv2.g_i_I[ist] = np.sum(ctv2.g_i_IJ[ist], axis=1)

            ctv2.g_I[:] = np.sum(ctv2.g_i_I, axis=0)
            ctv2.pseudo_pop = rho.T
            sigma_sq = ctv2.sigma ** 2
            inv_sigma_sq = np.where(sigma_sq > ctv2.small, 1.0 / sigma_sq, 0.0)
            ctv2.slope = -np.einsum('st,sab->tab', ctv2.pseudo_pop, inv_sigma_sq)

        # Warmup
        for _ in range(warmup):
            numpy_slope()

        # Benchmark NumPy
        times_numpy = []
        for _ in range(iterations):
            start = time.perf_counter()
            numpy_slope()
            times_numpy.append(time.perf_counter() - start)

        mean_numpy = np.mean(times_numpy) * 1000
        std_numpy = np.std(times_numpy) * 1000
        print(f"  NumPy:  {mean_numpy:.3f} +/- {std_numpy:.3f} ms")

        # GPU version
        if has_torch:
            reset_backend()
            gpu = get_backend('auto')
            kernels = CTv2GPUKernels(gpu)

            # Reset data
            ctv2_gpu = create_mock_ctv2_data(ntrajs)

            def gpu_slope():
                kernels.calculate_slope(ctv2_gpu)
                if gpu.device is not None and gpu.device.type in ('mps', 'cuda'):
                    gpu.synchronize()

            # Warmup
            for _ in range(warmup):
                gpu_slope()

            # Benchmark
            times_gpu = []
            for _ in range(iterations):
                start = time.perf_counter()
                gpu_slope()
                times_gpu.append(time.perf_counter() - start)

            mean_gpu = np.mean(times_gpu) * 1000
            std_gpu = np.std(times_gpu) * 1000
            speedup = mean_numpy / mean_gpu

            print(f"  GPU ({gpu.device_name}):  {mean_gpu:.3f} +/- {std_gpu:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")

            results.append({
                'ntrajs': ntrajs,
                'numpy_ms': mean_numpy,
                'gpu_ms': mean_gpu,
                'speedup': speedup,
                'device': gpu.device_name
            })

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark GPU acceleration for CTv2')
    parser.add_argument('--ntrajs', nargs='+', type=int, default=[100, 500, 1000],
                        help='List of trajectory counts to benchmark')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of benchmark iterations')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Number of warmup iterations')
    args = parser.parse_args()

    print("=" * 70)
    print("PyUNIxMD CTv2 GPU Acceleration Benchmark")
    print("=" * 70)

    # Print system info
    print("\nSystem Information:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  NumPy: {np.__version__}")

    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        if torch.backends.mps.is_available():
            print("  GPU: Apple Silicon (MPS)")
        elif torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name()}")
        else:
            print("  GPU: Not available")
    except ImportError:
        print("  PyTorch: Not installed")

    # Run benchmarks
    benchmark_gaussian_matrix(args.ntrajs, args.iterations, args.warmup)
    benchmark_full_slope(args.ntrajs, args.iterations, args.warmup)

    print("\n" + "=" * 70)
    print("Benchmark complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
