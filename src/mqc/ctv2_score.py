from __future__ import division
import numpy as np
from scipy.interpolate import BSpline

from misc import call_name
from mqc.ctv2 import CTv2

class CTv2Score(CTv2):
    """ CTv2 with a variational score-function estimator for the quantum momentum
        and the quantum potential (1D model systems only)

        The nuclear density enters the dynamics only through score-like ratios:
        the quantum momentum W = \\nabla|\\chi|^2/|\\chi|^2 (score of the total
        density), the state-pair quantum momentum W_i + W_j (scores of the
        BO-projected densities), and the quantum potential built from the total
        score. Here each score s(R) = d ln rho / dR is expanded in a cubic
        B-spline basis and fitted with the weak-form (Galerkin) identity
        <phi s> = -<phi'> over the trajectory ensemble -- sample means only, the
        data are never differentiated -- regularized by the smoothing-spline
        penalty tau * int s''^2. BO-projected densities use the same fit with
        rho_jj trajectory weights. This replaces the Gaussian-mixture density
        reconstruction (calculate_sigma/calculate_slope/calculate_center)
        entirely; K, K_bo, and the quantum potential force are assembled from
        the fitted scores with unchanged formulas.

        :param object,list molecules: List of molecule objects
        :param integer score_nknots: Number of interior knots of the B-spline basis
        :param double score_tau: Smoothing penalty strength tau (au)
        Remaining parameters are identical to CTv2. Not supported here:
        l_traj_gaussian, l_qpot_real_pop (mixture-model options), use_gpu.
    """
    def __init__(self, molecules, score_nknots=40, score_tau=1.0E-1, **kwargs):
        super().__init__(molecules=molecules, **kwargs)

        self.score_nknots = score_nknots
        self.score_tau = score_tau

        if (self.nat_qm != 1 or self.ndim != 1):
            error_message = "The score-function estimator is implemented for 1D model systems only!"
            error_vars = f"nat_qm = {self.nat_qm}, ndim = {self.ndim}"
            raise NotImplementedError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

        if (self.l_traj_gaussian):
            error_message = "l_traj_gaussian is a Gaussian-mixture option; not defined for the score estimator!"
            error_vars = f"l_traj_gaussian = {self.l_traj_gaussian}"
            raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

        if (self.l_qpot_real_pop):
            error_message = "l_qpot_real_pop needs mixture ratios; not defined for the score estimator!"
            error_vars = f"l_qpot_real_pop = {self.l_qpot_real_pop}"
            raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

        if (self.use_gpu):
            error_message = "GPU acceleration is not supported for the score estimator!"
            error_vars = f"use_gpu = {self.use_gpu}"
            raise NotImplementedError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

        # Penalty Gram matrix on a unit-span knot vector, computed once; for a
        # clamped uniform basis on span S it rescales exactly as P = P_unit / S^3
        self._p_unit = None
        self._sp_tot = None  # total-density score spline of the current step

    # ------------------------------------------------------------------
    # Weak-form B-spline score fit
    # ------------------------------------------------------------------
    def _knots(self, lo, hi):
        """ Clamped cubic knot vector with score_nknots uniform interior knots """
        t_int = np.linspace(lo, hi, self.score_nknots)
        t = np.concatenate([[lo] * 3, t_int, [hi] * 3])
        return t, len(t) - 4

    def _design(self, t, nb, pts, deriv=0):
        A = np.zeros((pts.size, nb))
        for j in range(nb):
            c = np.zeros(nb)
            c[j] = 1.
            A[:, j] = BSpline(t, c, 3)(pts, nu=deriv)
        return A

    def _penalty_unit(self):
        """ int phi_j'' phi_k'' dx on the unit-span basis (cached); phi'' is
            piecewise linear, so 2-point Gauss per knot interval is exact """
        if (self._p_unit is None):
            t, nb = self._knots(0., 1.)
            edges = np.unique(t)
            mid = 0.5 * (edges[:-1] + edges[1:])
            half = 0.5 * (edges[1:] - edges[:-1])
            gpts = np.concatenate([mid - half / np.sqrt(3.), mid + half / np.sqrt(3.)])
            gwts = np.concatenate([half, half])
            d2 = self._design(t, nb, gpts, deriv=2)
            self._p_unit = d2.T @ (d2 * gwts[:, np.newaxis])
        return self._p_unit

    def fit_score(self, x, w=None):
        """ Fit s = d ln rho / dR of the (w-weighted) trajectory density;
            returns the BSpline object """
        lo, hi = x.min(), x.max()
        span = hi - lo
        lo -= 0.05 * span
        hi += 0.05 * span
        t, nb = self._knots(lo, hi)

        phi = BSpline.design_matrix(x, t, 3).toarray()
        dphi = self._design(t, nb, x, deriv=1)
        if (w is None):
            wn = np.full(x.size, 1. / x.size)
        else:
            wn = np.asarray(w, dtype=float)
            wn = wn / np.sum(wn)
        gram = phi.T @ (phi * wn[:, np.newaxis])
        load = -(dphi * wn[:, np.newaxis]).sum(axis=0)
        pen = self._penalty_unit() / (hi - lo) ** 3

        try:
            c = np.linalg.solve(gram + self.score_tau * pen, load)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(gram + self.score_tau * pen, load, rcond=None)[0]
        return BSpline(t, c, 3)

    # ------------------------------------------------------------------
    # Quantum momentum from the fitted scores
    # ------------------------------------------------------------------
    def calculate_qmom(self):
        """ Routine to calculate quantum momentum from spline score fits
        """
        # i and j are state indices; I is the trajectory index.
        # a is the nucleus index and d is the cartesian component index.
        # -------------------------------------------------------------------
        # 1. Fit the total-density score (uniform weights over trajectories)
        # W = \nabla|\chi|^2 / |\chi|^2 = s_tot(R)
        pos = np.array([mol.pos[0, 0] for mol in self.mols])  # (ntrajs,)
        rho = np.array([np.diag(mol.rho.real) for mol in self.mols])  # (ntrajs, nst)
        rho_avg = np.sum(rho, axis=0) / self.ntrajs  # (nst,)
        valid_state = rho_avg >= self.lower_th  # (nst,)

        self._sp_tot = self.fit_score(pos)
        self.qmom[:, 0, 0] = self._sp_tot(pos)

        # 2. Fit the BO-projected density scores (rho_jj trajectory weights)
        # and assemble the state-pair quantum momentum W_i + W_j
        if (self.l_crunch):
            s_state = np.zeros((self.nst, self.ntrajs))
            for ist in range(self.nst):
                if (valid_state[ist]):
                    s_state[ist] = self.fit_score(pos, w=rho[:, ist])(pos)
            index_lk = 0
            for ist in range(self.nst):
                for jst in range(ist + 1, self.nst):
                    if (valid_state[ist] and valid_state[jst]):
                        self.qmom_bo[:, index_lk, 0, 0] = s_state[ist] + s_state[jst]
                    else:
                        self.qmom_bo[:, index_lk, 0, 0] = 0.0
                    index_lk += 1

        # 3. Calculate K and K_bo (identical to CTv2)
        # K_bo = 0.5 * G_{\nu, ij}/M \cdot D_{ij}
        # and/or
        # K = 0.5 * P_{\nu}/M \cdot D_{ij}
        self.K.fill(0.)
        self.K_bo.fill(0.)

        inv_mass = 1. / self.mol.mass[0:self.nat_qm]  # (nat_qm,)

        phase_diff = self.phase[:, :, np.newaxis, :, :] - self.phase[:, np.newaxis, :, :, :]
        coh_ij = self.l_coh[:, :, np.newaxis] & self.l_coh[:, np.newaxis, :]

        qmom_phase = np.einsum('Iad,Iijad->Iija', self.qmom, phase_diff)
        K_full = 0.5 * np.einsum('a,Iija->Iij', inv_mass, qmom_phase)

        triu_mask = np.triu(np.ones((self.nst, self.nst), dtype=bool), k=1)
        self.K = np.where(coh_ij & triu_mask, K_full, 0.0)
        self.K -= self.K.transpose(0, 2, 1)

        if (self.l_crunch):
            index_lk = 0
            for ist in range(self.nst):
                for jst in range(ist + 1, self.nst):
                    qmom_bo_phase = np.sum(self.qmom_bo[:, index_lk, :, :] * phase_diff[:, ist, jst, :, :], axis=2)
                    K_bo_val = 0.5 * np.sum(inv_mass * qmom_bo_phase, axis=1)  # (ntrajs,)
                    self.K_bo[:, ist, jst] = np.where(coh_ij[:, ist, jst], K_bo_val, 0.0)
                    self.K_bo[:, jst, ist] = -self.K_bo[:, ist, jst]
                    index_lk += 1

    # ------------------------------------------------------------------
    # Quantum potential from the fitted total-density score
    # ------------------------------------------------------------------
    def calculate_qpot_force(self):
        """ Routine to calculate the quantum potential and its nuclear force
            from the total-density score spline fitted in calculate_qmom

            With r = s/2 = \\nabla\\sqrt{rho}/\\sqrt{rho}:
            Q = -(1/2M)(r' + r^2), F^QP = -dQ/dR = (1/2M)(r'' + 2 r r')
        """
        pos = np.array([mol.pos[0, 0] for mol in self.mols])  # (ntrajs,)
        sp = self._sp_tot

        r = 0.5 * sp(pos)
        r1 = 0.5 * sp(pos, nu=1)
        r2 = 0.5 * sp(pos, nu=2)
        inv_2m = 0.5 / self.mol.mass[0]

        self.qpot = -(r1 + r ** 2) * inv_2m
        self.qpot_force[:, 0, 0] = (r2 + 2. * r * r1) * inv_2m

    def print_init(self, qm, mm, restart):
        """ Routine to print the initial information of dynamics
        """
        super().print_init(qm, mm, restart)

        score_info = "  Score-Function Estimator\n"
        score_info += f"  {'score_nknots':<27s}{'=':<5s}{self.score_nknots:>16d}\n"
        score_info += f"  {'score_tau':<27s}{'=':<5s}{self.score_tau:>16.6E}\n"
        print (score_info, flush=True)
