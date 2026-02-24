.. _Vectorization:

===========================
Vectorization Patterns
===========================

PyUNIxMD makes extensive use of NumPy vectorization throughout the codebase to eliminate
Python-level loops over atoms, states, and trajectories. This section documents the key
patterns organized by category.

Index conventions used below:

- ``i, j``: BO state indices (nst)
- ``a``: atom index (nat_qm)
- ``b, d``: spatial dimension index (ndim)
- ``k, l``: atom/dimension in force context
- ``t``: trajectory index (ntrajs, multi-trajectory methods)
- ``s``: BO state index in population context (nst)


Kinetic Energy (Molecule, Polariton)
--------------------------------------

Used in ``Molecule.update_kinetic()`` and ``Polariton.update_kinetic()`` to compute kinetic
energy without looping over atoms and dimensions:

.. code-block:: python

   # mass: (nat,), vel: (nat, ndim) → scalar
   self.ekin = 0.5 * np.einsum('i,ij->', self.mass, self.vel ** 2)

The mass-weighted sum of squared velocities replaces a double loop over atoms and dimensions.


NAC Phase Adjustment (Molecule, Polariton)
--------------------------------------------

``Molecule.adjust_nac()`` computes norms, dot products, and sign flips for all state pairs
simultaneously:

.. code-block:: python

   # nac: (nst, nst, nat_qm, ndim) — reduce over atom and dimension axes
   snac = np.sqrt(np.sum(self.nac ** 2, axis=(2, 3)))        # (nst, nst) norms
   dot_nac = np.sum(self.nac_old * self.nac, axis=(2, 3))    # (nst, nst) overlaps

   # Broadcast sign matrix to 4D NAC array
   self.nac *= sign_matrix[:, :, np.newaxis, np.newaxis]

The ``np.newaxis`` broadcasting applies a 2D sign correction to the full 4D NAC tensor
without iterating over state pairs.


NACME from NAC Vectors (Molecule, Polariton)
-----------------------------------------------

``Molecule.get_nacme()`` contracts the 4D NAC tensor with the velocity vector using
``np.tensordot``:

.. code-block:: python

   # nac: (nst, nst, nat_qm, ndim), vel_qm: (nat_qm, ndim) → (nst, nst)
   nacme_full = np.tensordot(self.nac, vel_qm, axes=([2, 3], [0, 1]))

   # Enforce antisymmetry
   self.nacme = np.triu(nacme_full, k=1)
   self.nacme -= self.nacme.T

This replaces nested loops over states, atoms, and dimensions with a single tensor contraction.
The same pattern appears in ``Polariton.get_nacme()`` and ``Polariton.get_pnacme()``.


Ehrenfest Force (Eh, CT, EhXF, QED variants)
-----------------------------------------------

The Ehrenfest force appears in multiple methods (``Eh``, ``CT``, ``CTv2``, and their QED counterparts).
The adiabatic and non-adiabatic contributions are vectorized as:

.. code-block:: python

   # Adiabatic force: population-weighted sum of state forces
   # rho_diag: (nst,), forces: (nst, nat, ndim)
   self.rforce = np.einsum('i,i...->...', rho_diag, forces)

   # Non-adiabatic force: energy-difference × coupling × coherence
   # energy_diff: (npairs,), rho_ij: (npairs,), nac_triu: (npairs, nat, ndim)
   self.rforce += np.einsum('k,k,k...->...', energy_diff, rho_ij, 2. * nac_triu)

The ``'i,i...->...'`` subscript broadcasts the population weights across arbitrary trailing
dimensions (supporting both 1D models and 3D molecular systems). The upper-triangle indexing
via ``np.triu_indices`` avoids double-counting of state pairs.


Velocity Rescaling Quadratic (SH, SHXF, SHXFv2)
---------------------------------------------------

Surface hopping methods solve a quadratic equation for the velocity rescaling factor
after a hop. The coefficients are computed via einsum in ``evaluate_hop()``:

.. code-block:: python

   # Velocity rescaling: a*gamma^2 + b*gamma + c = 0
   # mass: (nat,), nac: (nat, ndim), vel: (nat, ndim)

   # 'velocity' mode
   a = np.einsum('i,ij->', mass, nac ** 2)
   b = 2. * np.einsum('i,ij,ij->', mass, nac, vel)

   # 'momentum' / 'augment' mode
   a = np.einsum('i,ij->', 1. / mass, nac ** 2)
   b = 2. * np.einsum('ij,ij->', nac, vel)

These patterns appear identically in ``SH``, ``SHXF``, and ``SHXFv2``, computing the
mass-weighted NAC norm and NAC-velocity dot product.


XF Decoherence Force (EhXF)
-------------------------------

``EhXF.calculate_xf_force()`` computes the exact-factorization decoherence force from
phase differences across state pairs:

.. code-block:: python

   # weight: (nst, nst), phase_diff: (nst, nst, nat, ndim)
   self.xf_force[0:self.aux.nat] = np.einsum('ij,ijab->ab', weight, phase_diff)

This contracts the state-pair weight matrix with the phase gradient differences, reducing
from 4D to 2D (atom, dimension).


Coupled-Trajectory Force and Quantum Momentum (CT)
-----------------------------------------------------

The CT method uses einsum for the coupled-trajectory force term and the quantum momentum
kinetic energy:

.. code-block:: python

   # CT force: weight: (nst, nst), phase_diff: (nst, nst, nat, ndim)
   ctforce = np.einsum('ij,ijab->ab', weight, phase_diff) / (self.nst - 1)

   # Quantum momentum kinetic energy per state pair
   # mass_inv: (nat,), qmom_phase: (ntrajs, nst_pair, nat)
   K_ist = 2. * np.einsum('i,tpi->tp', mass_inv, qmom_phase_ist)

   # Gaussian-weighted population smoothing
   # prod_g_i: (ntrajs, ntrajs), rho_diag: (ntrajs, nst)
   self.w_k = np.einsum('ij,jk->ik', self.prod_g_i, rho_diag) / self.g_i[:, np.newaxis]

   # Weighted center from trajectory positions
   # weight: (ntrajs, ntrajs, nat, ndim), all_pos: (ntrajs, nat, ndim)
   center_new_base = np.einsum('ijad,jad->iad', weight, all_pos)


CTv2 Cross-Trajectory Calculations
--------------------------------------

CTv2 extends the CT patterns with fully batched trajectory-level vectorization.

**CT decoherence force** (replaces nested state-pair loops):

.. code-block:: python

   # K_diff: (nst, nst), phase_diff: (nst, nst, nat, ndim), rho_ij: (nst, nst)
   ctforce = -np.einsum('ij,ijkl,ij->kl', K_diff, phase_diff, rho_ij)

**K matrix** (fully vectorized across all trajectories):

.. code-block:: python

   # qmom: (ntrajs, nat, ndim), phase_diff: (ntrajs, nst, nst, nat, ndim)
   qmom_phase = np.einsum('tad,tijad->tija', self.qmom, phase_diff)

   # inv_mass: (nat,), qmom_phase: (ntrajs, nst, nst, nat)
   K_full = 0.5 * np.einsum('a,tija->tij', inv_mass, qmom_phase)

**Slope and intercept** (quantum momentum fitting):

.. code-block:: python

   # Slope: pseudo_pop: (nst, ntrajs), inv_sigma_sq: (nst, nat, ndim)
   self.slope = -np.einsum('st,sab->tab', self.pseudo_pop, inv_sigma_sq)

   # Intercept (trajectory-centered Gaussians):
   # g_i_IJ: (nst, ntrajs, ntrajs), pos: (ntrajs, nat, ndim), inv_sigma_sq: (nst, nat, ndim)
   intercept_sum = np.einsum('sij,jad,sad->iad', self.g_i_IJ, pos, inv_sigma_sq)

   # Intercept (single Gaussian):
   # pseudo_pop: (nst, ntrajs), weighted_avg_R: (nst, nat, ndim)
   self.intercept = -np.einsum('st,sab->tab', self.pseudo_pop, weighted_avg_R)

**Laplacian ENC term**:

.. code-block:: python

   # inv_mass: (nat,), d2S_diff: (nst, nst, nat), alpha_mat: (nst, nst, nat)
   lap_term = np.einsum('a,ija,ija->ij', inv_mass, d2S_diff, alpha_mat)

**Sigma calculation** (trajectory-weighted statistics via ``np.tensordot``):

.. code-block:: python

   # pos: (ntrajs, nat, ndim), rho: (ntrajs,) → (nat, ndim)
   self.avg_R[ist] = np.tensordot(pos, rho[:, ist], axes=(0, 0)) / rho_sum[ist]


Model System NAC (Shin-Metiu)
---------------------------------

The Shin-Metiu model computes nonadiabatic couplings from the derivative coupling matrix
using broadcasting to create the full energy difference matrix:

.. code-block:: python

   # ws: (nst,) eigenvalues → energy_diff: (nst, nst)
   energy_diff = ws[np.newaxis, :] - ws[:, np.newaxis]
   energy_diff_safe = np.where(energy_diff != 0, energy_diff, 1.)
   nac_full = dVijs / energy_diff_safe

This avoids explicit loops over state pairs and handles the diagonal zero-division safely.


QED Polarization (Jaynes-Cummings)
--------------------------------------

The Jaynes-Cummings model computes the light-matter coupling tensor via einsum:

.. code-block:: python

   # field_pol_vec: (ndim,), tdp: (pst, pst, ndim) → polarization: (pst, pst)
   polarization = np.einsum('k,ijk->ij', polariton.field_pol_vec,
                             polariton.tdp[ind_mol1, ind_mol2])

This contracts the field polarization vector with the transition dipole matrix
for all polariton state pairs.


Broadcasting Conventions
---------------------------

State-Pair Outer Products
'''''''''''''''''''''''''''''

Density matrix outer products for weighting state pairs:

.. code-block:: python

   # rho_diag: (nst,) → (nst, nst)
   rho_ij = rho_diag[:, np.newaxis] * rho_diag[np.newaxis, :]

Used in ``CT.calculate_force()``, ``CTv2.calculate_force()``, and ``CTv2.calculate_qmom()``
to weight contributions by population products.

Phase Differences
'''''''''''''''''''''''''''''

Multi-trajectory phase differences broadcast to high-dimensional tensors:

.. code-block:: python

   # phase: (ntrajs, nst, nat, ndim) → (ntrajs, nst, nst, nat, ndim)
   phase_diff = phase[:, :, np.newaxis, :, :] - phase[:, np.newaxis, :, :, :]

Used in ``CTv2.calculate_force()`` and ``CTv2.calculate_qmom()``.
For single-trajectory XF methods, the 4D equivalent is used without the trajectory axis.

Coherence Masks
'''''''''''''''''''''''''''''

Boolean masks broadcast for conditional state-pair calculations:

.. code-block:: python

   # l_coh: (ntrajs, nst) → (ntrajs, nst, nst)
   coh_ij = l_coh[:, :, np.newaxis] & l_coh[:, np.newaxis, :]

Combined with ``np.triu`` to select upper-triangle state pairs:

.. code-block:: python

   triu_mask = np.triu(np.ones((self.nst, self.nst), dtype=bool), k=1)
   self.K = np.where(coh_ij & triu_mask, K_full, 0.0)

Auxiliary Trajectory Initialization
''''''''''''''''''''''''''''''''''''''

Single-molecule geometry broadcast to multiple auxiliary states:

.. code-block:: python

   # mol_pos: (nat, ndim) → aux.pos: (nst, nat, ndim)
   self.aux.pos[:, :, :] = mol_pos[np.newaxis, :, :]

Used in ``SHXF``, ``SHXFv2``, and ``EhXF``.

Batch Data Extraction
'''''''''''''''''''''''''''''

Multi-trajectory methods stack per-molecule arrays into contiguous tensors:

.. code-block:: python

   pos = np.array([mol.pos for mol in self.mols])                    # (ntrajs, nat, ndim)
   rho = np.array([np.diag(mol.rho.real) for mol in self.mols])      # (ntrajs, nst)
   forces = np.array([st.force for st in self.mol.states])           # (nst, nat, ndim)
   energies = np.array([st.energy for st in self.mol.states])        # (nst,)

Used in ``CT``, ``CTv2``, ``Eh``, and their QED counterparts.


Matrix Symmetry Operations
-----------------------------

Antisymmetric matrices (NAC, NACME, K matrix) are enforced using ``np.triu``:

.. code-block:: python

   # Extract upper triangle and antisymmetrize
   self.nacme = np.triu(nacme_full, k=1)
   self.nacme -= self.nacme.T

   # 3D antisymmetrization (K matrix across trajectories)
   self.K -= self.K.transpose(0, 2, 1)

Hermitian density matrix reconstruction after surface hopping:

.. code-block:: python

   self.mol.rho = np.triu(self.mol.rho) + np.triu(self.mol.rho, k=1).conj().T


Safe Division with np.where
-------------------------------

Several methods use ``np.where`` for element-wise conditional operations that avoid
division by zero:

.. code-block:: python

   # NAC from derivative couplings (Shin-Metiu)
   energy_diff_safe = np.where(energy_diff != 0, energy_diff, 1.)
   nac_full = dVijs / energy_diff_safe

   # Inverse sigma (CTv2)
   inv_sigma_sq = np.where(sigma_sq > self.small, 1.0 / sigma_sq, 0.0)

   # Quantum momentum center (CTv2)
   self.center = np.where(slope_valid, self.intercept / slope_safe, pos)

   # Phase reset on decoherence (CT)
   self.phase[itraj] = np.where(outside_threshold[:, np.newaxis, np.newaxis],
                                 0., self.phase[itraj] + phase_update)
