
SHXFv2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Decoherence-induced surface hopping based on exact factorization (DISH-XF, or SHXFv2) :cite:`Ha2018,Kim2022`
is included in PyUNIxMD. SHXFv2 extends the SHXF method with the CRUNCH (coupled-trajectory using
non-adiabatic coupling history) decoherence scheme and phase correction options.

Like SHXF, the electronic equation of motion contains a decoherence term derived from exact factorization:

.. math::

   \dot C^{(I)}_k(t) =& -\frac{i}{\hbar}E^{(I)}_k(t)C^{(I)}_k(t)
   - \sum_j\sum_\nu{\bf d}^{(I)}_{kj\nu}(t)\cdot\dot{\bf R}^{(I)}_\nu(t)C^{(I)}_j(t) \nonumber\\
   &+\sum_j\sum_\nu\frac{1}{M_\nu}\frac{\nabla_\nu|\chi|}{|\chi|}\Bigg|_{\underline{\underline{\bf R}}^{(I)}(t)}
   \cdot\left\{{\bf f}^{(I)}_{j\nu}(t)-{\bf f}^{(I)}_{k\nu}(t)\right\}|C^{(I)}_j(t)|^2 C^{(I)}_k(t)

SHXFv2 differs from SHXF in:

- Support for CRUNCH decoherence (**l_crunch**) with projected quantum momentum
- Phase correction schemes (**t_pc**) with fine-grained control options
- Default **aux_econs_viol** is *'collapse'* (vs *'fix'* in SHXF)
- Option to disable hopping (**l_no_hop**) for pure decoherence studies

+----------------------------+------------------------------------------------------+--------------+
| Parameters                 | Work                                                 | Default      |
+============================+======================================================+==============+
| **molecule**               | Molecule object                                      |              |
| (:class:`Molecule`)        |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **thermostat**             | Thermostat object                                    | *None*       |
| (:class:`Thermostat`)      |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **istate**                 | Initial state                                        | *0*          |
| *(integer)*                |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **dt**                     | Time interval                                        | *0.5*        |
| *(double)*                 |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **nsteps**                 | Total step of nuclear propagation                    | *1000*       |
| *(integer)*                |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **nesteps**                | Total step of electronic propagation                 | *20*         |
| *(integer)*                |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **elec_object**            | Electronic equation of motions                       |*'coefficient'*|
| *(string)*                 |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **propagator**             | Electronic propagator                                | *'rk4'*      |
| *(string)*                 |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **l_print_dm**             | Logical to print BO population and coherence         | *True*       |
| *(boolean)*                |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **l_adj_nac**              | Adjust nonadiabatic coupling to align phases         | *True*       |
| *(boolean)*                |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **hop_rescale**            | Velocity rescaling method after successful hop       | *'augment'*  |
| *(string)*                 |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **hop_reject**             | Velocity rescaling method after frustrated hop       | *'reverse'*  |
| *(string)*                 |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **rho_threshold**          | Electronic density threshold for decoherence term    | *0.01*       |
| *(double)*                 | calculation                                          |              |
+----------------------------+------------------------------------------------------+--------------+
| **sigma**                  | Width of nuclear wave packet of auxiliary trajectory  | *None*       |
| *(double/(double, list))*  |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **init_coef**              | Initial BO coefficient                               | *None*       |
| *(double/complex, list)*   |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **l_econs_state**          | Logical to use identical total energies              | *True*       |
| *(boolean)*                | for all auxiliary trajectories                        |              |
+----------------------------+------------------------------------------------------+--------------+
| **aux_econs_viol**         | How to treat trajectories violating total energy     | *'fix'*      |
| *(string)*                 | conservation                                         |              |
+----------------------------+------------------------------------------------------+--------------+
| **l_crunch**               | Use CRUNCH decoherence                               | *True*       |
| *(boolean)*                |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **t_pc**                   | Phase correction scheme                              | *0*          |
| *(integer)*                |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **l_exact_pc**             | Use exact phase correction                           | *True*       |
| *(boolean)*                |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **l_pc_w_phase_term**      | Use phase term in phase correction                   | *False*      |
| *(boolean)*                |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **l_divide_rho**           | Divide by rho in decoherence term                    | *False*      |
| *(boolean)*                |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **l_pc_nac**               | Use NAC in phase correction                          | *False*      |
| *(boolean)*                |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **l_pc_rescale**           | Rescale after phase correction                       | *False*      |
| *(boolean)*                |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **l_no_hop**               | Disable hopping                                      | *False*      |
| *(boolean)*                |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **l_asymp**                | Terminate dynamics in asymptotic region              | *False*      |
| *(boolean)*                | (model systems only)                                 |              |
+----------------------------+------------------------------------------------------+--------------+
| **x_fin**                  | Asymptotic region boundary (a.u.)                    | *25.0*       |
| *(double)*                 |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **unit_dt**                | Unit of time interval                                | *'fs'*       |
| *(string)*                 |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **out_freq**               | Frequency of printing output                         | *1*          |
| *(integer)*                |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+
| **verbosity**              | Verbosity of output                                  | *0*          |
| *(integer)*                |                                                      |              |
+----------------------------+------------------------------------------------------+--------------+

Detailed description of the parameters
""""""""""""""""""""""""""""""""""""""""

- **istate** *(integer)* - Default: *0* (Ground state)

  This parameter specifies the initial running state. The possible range is from *0* to ``molecule.nst - 1``.

\

- **dt** *(double)* - Default: *0.5*

  This parameter determines the time interval of the nuclear time steps.
  You can select the unit of time for the dynamics with the **unit_dt** parameter.

\

- **nsteps** *(integer)* - Default: *1000*

  This parameter determines the total number of the nuclear time steps.

\

- **nesteps** *(integer)* - Default: *20*

  This parameter determines the number of electronic time steps between one nuclear time step for the integration of the electronic equation of motion.

\

- **elec_object** *(string)* - Default: *'coefficient'*

  The **elec_object** parameter determines the representation for the electronic state.

  + *'density'*: Propagates the density matrix elements, i.e., :math:`\{\rho_{ij}^{(I)}(t)\}`
  + *'coefficient'*: Propagates the coefficients, i.e., :math:`\{C_{i}^{(I)}(t)\}`

\

- **propagator** *(string)* - Default: *'rk4'*

  This parameter determines the numerical integration method for the electronic equation of motion.
  Currently, only the RK4 algorithm (*'rk4'*) is available.

\

- **l_print_dm** *(boolean)* - Default: *True*

  This parameter determines whether to write output files for density matrix elements ('BOPOP', 'BOCOH').

\

- **l_adj_nac** *(boolean)* - Default: *True*

  If this parameter is set to *True*, the signs of the NACVs are adjusted to match the phases to the previous time step during the dynamics.

\

- **hop_rescale** *(string)* - Default: *'augment'*

  This parameter determines the direction of the momentum to be adjusted after a hop to conserve the total energy.

  + *'energy'*: Simply rescale the nuclear velocities.
  + *'velocity'*: Adjust the velocity along the NACV direction.
  + *'momentum'*: Adjust the momentum in the direction of the NACV.
  + *'augment'*: First, the hop is evaluated as *'momentum'*.
    If the kinetic energy is not enough, then the hop is evaluated again as *'energy'*.

\

- **hop_reject** *(string)* - Default: *'reverse'*

  This parameter determines the momentum rescaling method when a hop is rejected.

  + *'keep'*: Do nothing, keeps the nuclear velocities.
  + *'reverse'*: Reverse the momentum along the NACV.

\

- **rho_threshold** *(double)* - Default: *0.01*

  This parameter defines the numerical threshold for the coherence.
  If the populations of two or more states are larger than this value, the electronic state is 'coherent' and the decoherence term is calculated.

\

- **sigma** *(double/(double, list))* - Default: *None* **(required)**

  This parameter defines the width (:math:`\sigma_\nu`) of the frozen Gaussian nuclear densities on the auxiliary trajectories.
  If a scalar value is given, all nuclei share the same width.
  If a list of values with the length of the number of atoms is given, atom-wise widths are used.

  .. note:: This parameter must be set explicitly; there is no default value.

\

- **init_coef** *(double/complex, list)* - Default: *None*

  This parameter defines the initial BO coefficients.
  The length should be equal to ``molecule.nst``.
  If not given, the BO coefficients are initialized according to **istate**.

\

- **l_econs_state** *(boolean)* - Default: *True*

  This parameter determines whether the total energies of all auxiliary trajectories are identical.
  If *True*, all auxiliary trajectories share the same total energy.

\

- **aux_econs_viol** *(string)* - Default: *'fix'*

  This parameter determines how to deal with auxiliary trajectories violating total energy conservation.

  + *'fix'*: Fix the auxiliary trajectory until decoherence.
  + *'collapse'*: Destroy the auxiliary trajectory, collapse the corresponding coefficient to zero, and renormalize.

\

- **l_crunch** *(boolean)* - Default: *True*

  If set to *True*, the CRUNCH projected quantum momentum is used for the decoherence term calculation.
  This provides state-pair-resolved decoherence, improving accuracy over the total quantum momentum.

\

- **t_pc** *(integer)* - Default: *0*

  Determines the phase correction scheme:

  + *0*: No phase correction
  + *1*: Phase correction using momentum
  + *2*: Phase correction using :math:`\sum_j \nabla S_j`

\

- **l_exact_pc** *(boolean)* - Default: *True*

  If set to *True*, use exact phase correction formula. Only effective when **t_pc** > 0.

\

- **l_pc_w_phase_term** *(boolean)* - Default: *False*

  If set to *True*, include the phase term in the phase correction calculation.

\

- **l_divide_rho** *(boolean)* - Default: *False*

  If set to *True*, divide by rho in the decoherence term expression.

\

- **l_pc_nac** *(boolean)* - Default: *False*

  If set to *True*, use nonadiabatic coupling in the phase correction.

\

- **l_pc_rescale** *(boolean)* - Default: *False*

  If set to *True*, rescale coefficients after applying phase correction.

\

- **l_no_hop** *(boolean)* - Default: *False*

  If set to *True*, surface hopping is disabled entirely. This is useful for studying
  pure decoherence dynamics without stochastic hopping events.

\

- **l_asymp** *(boolean)* - Default: *False*

  If set to *True*, the dynamics is terminated when the trajectory reaches the asymptotic region
  defined by **x_fin**. Intended for model systems only.

\

- **x_fin** *(double)* - Default: *25.0*

  Defines the asymptotic region boundary in atomic units.

\

- **unit_dt** *(string)* - Default: *'fs'*

  This parameter determines the unit of time for the simulation.

  + *'fs'*: Femtosecond
  + *'au'*: Atomic unit

\

- **out_freq** *(integer)* - Default: *1*

  PyUNIxMD prints and writes the dynamics information at every **out_freq** time step.

\

- **verbosity** *(integer)* - Default: *0*

  This parameter determines the verbosity of the output files and stream.

  + **verbosity** :math:`\geq` *1*: Prints accumulated hopping probabilities and random numbers,
    and writes decoherence terms in time-derivative of BO populations to DOTPOPDEC.
  + **verbosity** :math:`\geq` *2*: Writes the NACVs ('NACV\_\ :math:`i`\_\ :math:`j`'),
    quantum momentum ('QMOM\_\ :math:`i`\_\ :math:`j`'), phase terms ('AUX_PHASE\_\ :math:`i`'),
    and atomic positions and velocities of the auxiliary trajectories ('AUX_MOVIE\_\ :math:`i`.xyz')
    where :math:`i` and :math:`j` represent BO states.


Auxiliary Trajectories
""""""""""""""""""""""""""""""""""""""""

SHXFv2 uses auxiliary trajectories to construct the nuclear density gradient needed for the
decoherence term. Each BO state has an associated auxiliary trajectory with position and velocity
defined by the :class:`Auxiliary_Molecule` class.

The velocity of an auxiliary trajectory is given as the velocity of the true nuclear trajectory
multiplied by a scaling factor (alpha) determined from total energy conservation:

.. math::

   \underline{\underline{\dot{\textbf{R}}}}_{k} = \alpha_k \cdot \underline{\underline{\dot{\textbf{R}}}}^{(I)}, \quad
   \alpha_k = \sqrt{\dfrac{E_{tot}^k - E_k^{(I)}}{\sum_\nu \frac{1}{2}M_\nu|\dot{\textbf{R}}^{(I)}_\nu|^2}}

When :math:`E_{tot}^k - E_k^{(I)} < 0`, the auxiliary trajectory violates energy conservation
and is treated according to **aux_econs_viol** (*'fix'* or *'collapse'*).


Hopping Logic
""""""""""""""""""""""""""""""""""""""""

SHXFv2 uses a modified fewest-switches algorithm:

1. **hop_prob**: Calculates hopping probabilities using the fewest-switches formula.
   Additionally implements a *force hop* mechanism — when a state becomes fully
   decohered (population collapses to 0 or 1), a deterministic hop to the dominant state is triggered.

2. **hop_check**: Compares the accumulated probability against a random number to determine if a hop occurs.

3. **evaluate_hop**: After a hop is accepted, the nuclear velocity is rescaled to conserve
   total energy. For momentum/augment rescaling, a quadratic equation is solved to find the
   velocity adjustment along the NACV direction.


Decoherence Lifecycle
""""""""""""""""""""""""""""""""""""""""

The decoherence mechanism in SHXFv2 follows a lifecycle:

1. **check_coherence**: Monitors electronic populations. When a state's population exceeds
   **rho_threshold**, the state enters coherence (``l_coh[ist] = True``).

2. **check_decoherence**: When a state's population drops below **rho_threshold** while coherent,
   the state decoheres (``l_coh[ist] = False``). The auxiliary trajectory and phase are reset.

3. **set_decoherence**: Applies decoherence by collapsing the coefficient of the decohered state
   and renormalizing the remaining coefficients.


Output Files
""""""""""""""""""""""""""""""""""""""""

+-------------------------+-------------------------------------------------------+
| File                    | Description                                           |
+=========================+=======================================================+
| MOVIE.xyz               | Trajectory with energies in comment lines             |
+-------------------------+-------------------------------------------------------+
| FINAL.xyz               | Final geometry                                        |
+-------------------------+-------------------------------------------------------+
| DENSITY                 | Electronic state populations and coherences           |
+-------------------------+-------------------------------------------------------+
| NACME                   | Nonadiabatic coupling matrix elements                 |
+-------------------------+-------------------------------------------------------+
| SHSTATE                 | Running state at each time step                       |
+-------------------------+-------------------------------------------------------+
| SHPROB                  | Hopping probabilities at each time step               |
+-------------------------+-------------------------------------------------------+
| DOTPOPDEC               | Decoherence contributions (verbosity >= 1)            |
+-------------------------+-------------------------------------------------------+
| QMOM\_\ *i*\_\ *j*     | Quantum momentum (verbosity >= 2)                     |
+-------------------------+-------------------------------------------------------+
| AUX_PHASE\_\ *i*        | Phase term for state *i* (verbosity >= 2)             |
+-------------------------+-------------------------------------------------------+
| AUX_MOVIE\_\ *i*.xyz    | Auxiliary trajectory for state *i* (verbosity >= 2)   |
+-------------------------+-------------------------------------------------------+
| NACV\_\ *i*\_\ *j*      | NAC vectors (verbosity >= 2)                          |
+-------------------------+-------------------------------------------------------+
