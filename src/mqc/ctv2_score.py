from __future__ import division
import os
import numpy as np
from scipy.interpolate import BSpline

from misc import call_name, typewriter
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

        The penalty strength can be selected from the data instead of set by
        hand: score_tau = "cv" runs a k-fold held-out selection of the
        Hyvarinen objective J = mean(0.5 s^2 + s') -- no exact reference
        enters anywhere -- over a log-spaced candidate grid (widened
        automatically while the minimum sits at an endpoint); "gcv" uses an
        effective-degrees-of-freedom-corrected in-sample J on the same
        quadratic form (also the automatic fallback when the weighted folds
        degenerate). Selection runs per fit (total density and each
        BO-projected density), every score_cv_update steps, damped by the
        score_cv_max_factor hysteresis cap. The selected-tau trace (SCORE_TAU)
        and the held-out J curves (SCORE_TAU_CURVE) are written to the run
        output directory. See work/handoff/note_gb_tau_selection.pdf for the
        full derivation and algorithm.

        :param object,list molecules: List of molecule objects
        :param integer score_nknots: Number of interior knots of the B-spline basis
        :param score_tau: Smoothing penalty strength tau (au), or "cv" for
            k-fold held-out Hyvarinen selection, or "gcv" for the
            effective-degrees-of-freedom criterion
        :param score_pad: Knot-domain padding as a fraction of the data span;
            a scalar pads both sides, a (lo, hi) pair pads asymmetrically
        :param integer score_cv_kfold: Number of folds k for score_tau = "cv"
        :param integer score_cv_update: Re-select tau every this many steps
            (per fit); between updates the stored value is reused
        :param double score_cv_max_factor: Hysteresis cap; per update the
            selected tau may move at most by this factor (None disables)
        :param score_floor_neff: Occupancy floor: when the Kish effective
            sample size of the fit weights drops below this, the spline fit
            (a near-singular Gram) is replaced by the tau -> infinity limit,
            the moment-matched linear (Gaussian) score. "auto" = the number
            of basis functions (score_nknots + 2); None disables
        Remaining parameters are identical to CTv2. Not supported here:
        l_traj_gaussian, l_qpot_real_pop (mixture-model options), use_gpu.
    """
    # Candidate grid for tau selection: half-decade spacing over the current
    # empirical range; widened in half-decade steps within the hard bounds
    # while the running minimum sits at a grid endpoint
    _CV_GRID = np.logspace(-6., 0., 13)
    _CV_TAU_MIN = 1.0E-10
    _CV_TAU_MAX = 1.0E+4
    _CV_WIDEN_FACTOR = 10. ** 0.5

    def __init__(self, molecules, score_nknots=40, score_tau=1.0E-1, score_pad=0.05, \
        score_cv_kfold=5, score_cv_update=1, score_cv_max_factor=1.0E+1, \
        score_floor_neff="auto", **kwargs):
        super().__init__(molecules=molecules, **kwargs)

        self.score_nknots = score_nknots
        self.score_tau = score_tau

        if (isinstance(score_floor_neff, str)):
            if (score_floor_neff.lower() != "auto"):
                error_message = "String-valued score_floor_neff must be 'auto'!"
                error_vars = f"score_floor_neff = {score_floor_neff}"
                raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")
            self.score_floor_neff = float(score_nknots + 2)
        elif (score_floor_neff is None):
            self.score_floor_neff = None
        else:
            self.score_floor_neff = float(score_floor_neff)
        self.score_floor_count = {}

        if (isinstance(score_tau, str)):
            self.score_cv_mode = score_tau.lower()
            if (self.score_cv_mode not in ("cv", "gcv")):
                error_message = "String-valued score_tau must be 'cv' or 'gcv'!"
                error_vars = f"score_tau = {score_tau}"
                raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")
        else:
            self.score_cv_mode = None
            self.score_tau = float(score_tau)

        try:
            pad_lo, pad_hi = score_pad
        except TypeError:
            pad_lo = pad_hi = score_pad
        self.score_pad = (float(pad_lo), float(pad_hi))

        self.score_cv_kfold = score_cv_kfold
        self.score_cv_update = score_cv_update
        self.score_cv_max_factor = score_cv_max_factor

        if (self.score_cv_mode == "cv" and self.score_cv_kfold < 2):
            error_message = "Held-out selection needs at least two folds!"
            error_vars = f"score_cv_kfold = {score_cv_kfold}"
            raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

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

        # tau-selection state (score_tau = "cv"/"gcv"): per-fit selected tau,
        # per-fit call counters, selection counts, taus used in the current
        # step, and J curves not yet written to SCORE_TAU_CURVE
        self._tau_sel = {}
        self._sel_calls = {}
        self.score_cv_nsel = {}
        self.score_tau_used = {}
        self._cv_curves = []
        # calculate_qmom call counter; the first call belongs to istep = -1
        # (restart runs are not supported by the trace, as in CTv2 itself)
        self._score_step = -2
        self._score_out_dir = None

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

    def _solve(self, a, b):
        try:
            return np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(a, b, rcond=None)[0]

    def fit_score(self, x, w=None, tau=None, label=None):
        """ Fit s = d ln rho / dR of the (w-weighted) trajectory density;
            returns the BSpline object

            tau overrides the penalty for this fit. With score_tau = "cv" or
            "gcv" and tau = None the penalty is selected from the data; label
            keys the cadence/hysteresis state of the selection (label = None
            selects statelessly, e.g. for external one-shot calls)
        """
        lo, hi = x.min(), x.max()
        span = hi - lo
        lo -= self.score_pad[0] * span
        hi += self.score_pad[1] * span
        t, nb = self._knots(lo, hi)

        if (w is None):
            wr = np.ones(x.size)
        else:
            wr = np.asarray(w, dtype=float)

        # Occupancy floor: with fewer effective samples than basis functions
        # the Gram is rank-deficient and the spline coefficients are set by
        # the penalty and noise (the near-singular case the D1 note warns
        # about); fall back to the tau -> infinity limit of the same weak
        # form, the moment-matched linear (Gaussian) score
        if (self.score_floor_neff is not None):
            n_eff = np.sum(wr) ** 2 / np.sum(wr ** 2)
            if (n_eff < self.score_floor_neff):
                if (label is not None):
                    self.score_tau_used[label] = np.inf
                    self.score_floor_count[label] = self.score_floor_count.get(label, 0) + 1
                return self._linear_score(x, wr, t, nb)

        phi = BSpline.design_matrix(x, t, 3).toarray()
        dphi = self._design(t, nb, x, deriv=1)
        pen = self._penalty_unit() / (hi - lo) ** 3

        if (tau is None):
            if (self.score_cv_mode is None):
                tau = self.score_tau
            else:
                tau = self._tau_update(label, phi, dphi, wr, pen)

        wn = wr / np.sum(wr)
        gram = phi.T @ (phi * wn[:, np.newaxis])
        load = -(dphi * wn[:, np.newaxis]).sum(axis=0)
        c = self._solve(gram + tau * pen, load)
        return BSpline(t, c, 3)

    def _linear_score(self, x, w, t, nb):
        """ Moment-matched linear (Gaussian) score s = (mu - R) / var on the
            weighted ensemble -- the solution of the weak form restricted to
            the penalty null space {1, R} -- represented exactly in the cubic
            B-spline basis via the Greville abscissae """
        wn = w / np.sum(w)
        mu = np.sum(wn * x)
        var = np.sum(wn * (x - mu) ** 2)
        grev = (t[1:nb + 1] + t[2:nb + 2] + t[3:nb + 3]) / 3.
        if (var > 0.):
            c = (mu - grev) / var
        else:
            c = np.zeros(nb)
        return BSpline(t, c, 3)

    # ------------------------------------------------------------------
    # Data-driven selection of the penalty strength (score_tau = "cv"/"gcv")
    # ------------------------------------------------------------------
    def _tau_update(self, label, phi, dphi, w, pen):
        """ Return the tau to use for this fit: re-select at the
            score_cv_update cadence (per fit label), damp with the hysteresis
            cap, and queue the evaluated J curve for the trace files.
            label = None runs a fresh stateless selection. """
        if (label is not None):
            ncall = self._sel_calls.get(label, 0)
            self._sel_calls[label] = ncall + 1
            if (label in self._tau_sel and ncall % max(1, self.score_cv_update) != 0):
                self.score_tau_used[label] = self._tau_sel[label]
                return self._tau_sel[label]

        tau_new, curve = self._select_tau(phi, dphi, w, pen)

        if (label is None):
            self.score_cv_last_curve = curve
            return tau_new

        if (label in self._tau_sel and self.score_cv_max_factor is not None):
            f = self.score_cv_max_factor
            tau_new = float(np.clip(tau_new, self._tau_sel[label] / f, self._tau_sel[label] * f))

        self._tau_sel[label] = tau_new
        self.score_tau_used[label] = tau_new
        self.score_cv_nsel[label] = self.score_cv_nsel.get(label, 0) + 1
        self._cv_curves.append((self._score_step, label, curve))
        return tau_new

    def _select_tau(self, phi, dphi, w, pen):
        """ Scan tau over the log-spaced candidate grid, widening in
            half-decade steps while the minimum sits at an endpoint, and
            return (tau, curve) with curve the evaluated (tau, J) pairs """
        crit = self._make_criterion(phi, dphi, w, pen)

        def safe(tau):
            val = crit(tau)
            return float(val) if np.isfinite(val) else np.inf

        taus = list(self._CV_GRID)
        js = [safe(tau) for tau in taus]
        for _ in range(16):
            imin = int(np.argmin(js))
            if (imin == 0 and taus[0] > self._CV_TAU_MIN * self._CV_WIDEN_FACTOR):
                taus.insert(0, taus[0] / self._CV_WIDEN_FACTOR)
                js.insert(0, safe(taus[0]))
            elif (imin == len(taus) - 1 and taus[-1] < self._CV_TAU_MAX / self._CV_WIDEN_FACTOR):
                taus.append(taus[-1] * self._CV_WIDEN_FACTOR)
                js.append(safe(taus[-1]))
            else:
                break

        curve = list(zip(taus, js))
        if (not np.isfinite(np.min(js))):
            # every candidate failed (degenerate ensemble); fall back to the
            # geometric midpoint of the scanned range rather than an endpoint
            return float(np.sqrt(taus[0] * taus[-1])), curve
        return taus[int(np.argmin(js))], curve

    def _make_criterion(self, phi, dphi, w, pen):
        """ Build the scalar criterion J(tau) on the assembled design.

            "cv": split trajectories into k interleaved folds (deterministic;
            no RNG state is consumed, so seed-fixed runs stay reproducible);
            for each tau fit on k-1 folds with the same Galerkin solve and
            average the weight-normalized Hyvarinen objective
            J = mean(0.5 s^2 + s') over the held-out folds. Falls back to
            "gcv" if fewer than two folds carry weight.

            "gcv": in-sample J plus the Takeuchi optimism correction
            df(tau) / n_eff with df = tr[V (G + tau P)^-1], V the weighted
            covariance of the per-trajectory loss gradient (the effective
            degrees of freedom of the penalized fit; for a well-specified
            quadratic problem it reduces to the classical tr[G (G+tau P)^-1])
            and n_eff the Kish effective sample size of the weights. The
            gradient-covariance form is essential here: the naive df ignores
            the variance blow-up of the Hyvarinen loss for wiggly fits and
            systematically under-penalizes small tau. """
        wsum = np.sum(w)

        if (self.score_cv_mode == "cv"):
            k = self.score_cv_kfold
            fold = np.arange(phi.shape[0]) % k
            g_f = np.empty((k, phi.shape[1], phi.shape[1]))
            l_f = np.empty((k, phi.shape[1]))
            w_f = np.empty(k)
            for f in range(k):
                m = fold == f
                wf = w[m]
                g_f[f] = phi[m].T @ (phi[m] * wf[:, np.newaxis])
                l_f[f] = -(dphi[m] * wf[:, np.newaxis]).sum(axis=0)
                w_f[f] = np.sum(wf)
            usable = w_f > 1.0E-12 * wsum
            if (np.count_nonzero(usable) >= 2):
                g_tot = g_f.sum(axis=0)
                l_tot = l_f.sum(axis=0)

                def crit(tau):
                    js = []
                    for f in range(k):
                        w_tr = wsum - w_f[f]
                        if (not usable[f] or w_tr <= 0.):
                            continue
                        c = self._solve((g_tot - g_f[f]) / w_tr + tau * pen, (l_tot - l_f[f]) / w_tr)
                        js.append((0.5 * c @ g_f[f] @ c - l_f[f] @ c) / w_f[f])
                    return np.mean(js) if js else np.nan
                return crit
            # degenerate folds: fall through to the effective-dof criterion

        wn = w / wsum
        gram = phi.T @ (phi * wn[:, np.newaxis])
        load = -(dphi * wn[:, np.newaxis]).sum(axis=0)
        n_eff = wsum ** 2 / np.sum(w ** 2)

        def crit(tau):
            a = gram + tau * pen
            c = self._solve(a, load)
            # per-trajectory gradient of the Hyvarinen loss at the fit:
            # g_I = phi_I (phi_I . c) + dphi_I
            s_val = phi @ c
            gmat = phi * s_val[:, np.newaxis] + dphi
            gbar = (gmat * wn[:, np.newaxis]).sum(axis=0)
            v = (gmat * wn[:, np.newaxis]).T @ gmat - np.outer(gbar, gbar)
            df = np.trace(self._solve(a, v))
            return 0.5 * c @ gram @ c - load @ c + df / n_eff
        return crit

    def _write_tau_trace(self):
        """ Append the taus used this step (SCORE_TAU) and any newly evaluated
            J curves (SCORE_TAU_CURVE) to the run output directory; column
            layout: step, tau of the total-density fit, tau per BO-projected
            fit (nan where the fit was skipped this step) """
        curves, self._cv_curves = self._cv_curves, []
        if (self._score_out_dir is None):
            return

        labels = ["tot"] + [f"state_{ist}" for ist in range(self.nst)]
        if (self._score_step == -1):
            header = "#" + "step".rjust(7) + "".join(f"tau_{lab}".rjust(16) for lab in labels)
            typewriter(header, self._score_out_dir, "SCORE_TAU", "w")
            header = "#" + "step".rjust(7) + "label".rjust(11) + "ntau".rjust(5) + "   (tau, J) pairs"
            typewriter(header, self._score_out_dir, "SCORE_TAU_CURVE", "w")

        tmp = f"{self._score_step:8d}"
        for lab in labels:
            tmp += f"{self.score_tau_used.get(lab, np.nan):16.6E}"
        typewriter(tmp, self._score_out_dir, "SCORE_TAU", "a")

        for step, lab, curve in curves:
            tmp = f"{step:8d}{lab:>11s}{len(curve):5d}"
            for tau, jval in curve:
                tmp += f"{tau:14.4E}{jval:16.6E}"
            typewriter(tmp, self._score_out_dir, "SCORE_TAU_CURVE", "a")

    def run(self, qm, mm=None, output_dir="./", **kwargs):
        """ Wrap CTv2.run to record the output directory for the
            selected-tau trace files """
        if (self.score_cv_mode is not None):
            self._score_out_dir = os.path.join(os.getcwd(), output_dir)
        return super().run(qm, mm=mm, output_dir=output_dir, **kwargs)

    # ------------------------------------------------------------------
    # Quantum momentum from the fitted scores
    # ------------------------------------------------------------------
    def calculate_qmom(self):
        """ Routine to calculate quantum momentum from spline score fits
        """
        # i and j are state indices; I is the trajectory index.
        # a is the nucleus index and d is the cartesian component index.
        # -------------------------------------------------------------------
        self._score_step += 1
        self.score_tau_used = {}

        # 1. Fit the total-density score (uniform weights over trajectories)
        # W = \nabla|\chi|^2 / |\chi|^2 = s_tot(R)
        pos = np.array([mol.pos[0, 0] for mol in self.mols])  # (ntrajs,)
        rho = np.array([np.diag(mol.rho.real) for mol in self.mols])  # (ntrajs, nst)
        rho_avg = np.sum(rho, axis=0) / self.ntrajs  # (nst,)
        valid_state = rho_avg >= self.lower_th  # (nst,)

        self._sp_tot = self.fit_score(pos, label="tot")
        self.qmom[:, 0, 0] = self._sp_tot(pos)

        # 2. Fit the BO-projected density scores (rho_jj trajectory weights)
        # and assemble the state-pair quantum momentum W_i + W_j
        if (self.l_crunch):
            s_state = np.zeros((self.nst, self.ntrajs))
            for ist in range(self.nst):
                if (valid_state[ist]):
                    s_state[ist] = self.fit_score(pos, w=rho[:, ist], label=f"state_{ist}")(pos)
            index_lk = 0
            for ist in range(self.nst):
                for jst in range(ist + 1, self.nst):
                    if (valid_state[ist] and valid_state[jst]):
                        self.qmom_bo[:, index_lk, 0, 0] = s_state[ist] + s_state[jst]
                    else:
                        self.qmom_bo[:, index_lk, 0, 0] = 0.0
                    index_lk += 1

        if (self.score_cv_mode is not None):
            self._write_tau_trace()

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
        if (self.score_cv_mode is None):
            score_info += f"  {'score_tau':<27s}{'=':<5s}{self.score_tau:>16.6E}\n"
        else:
            score_info += f"  {'score_tau':<27s}{'=':<5s}{self.score_cv_mode:>16s}\n"
            score_info += f"  {'score_cv_kfold':<27s}{'=':<5s}{self.score_cv_kfold:>16d}\n"
            score_info += f"  {'score_cv_update':<27s}{'=':<5s}{self.score_cv_update:>16d}\n"
            if (self.score_cv_max_factor is None):
                score_info += f"  {'score_cv_max_factor':<27s}{'=':<5s}{'None':>16s}\n"
            else:
                score_info += f"  {'score_cv_max_factor':<27s}{'=':<5s}{self.score_cv_max_factor:>16.6E}\n"
        if (self.score_floor_neff is None):
            score_info += f"  {'score_floor_neff':<27s}{'=':<5s}{'None':>16s}\n"
        else:
            score_info += f"  {'score_floor_neff':<27s}{'=':<5s}{self.score_floor_neff:>16.1f}\n"
        if (self.score_pad != (0.05, 0.05)):
            score_info += f"  {'score_pad':<27s}{'=':<5s}{str(self.score_pad):>16s}\n"
        print (score_info, flush=True)
