
CTv2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coupled-trajectory mixed quantum-classical dynamics version 2 (CTv2) is an enhanced implementation
of the CTMQC method :cite:`Agostini2016` in PyUNIxMD. CTv2 introduces the CRUNCH (coupled-trajectory
using non-adiabatic coupling history) projected quantum momentum :cite:`Kim2022`, state-wise momentum
for phase calculations, population conservation schemes, and optional GPU acceleration and CPU parallelization.

The nuclear force in CTv2 consists of the Ehrenfest force plus a coupled-trajectory decoherence correction:

.. math::

   \mathbf{F}_{\nu}^{(I)}=-\sum_{i}|C_{i}^{(I)}|^2\nabla_{\nu}E_{i}^{(I)} + \sum_{i\neq j} C_{i}^{(I)\ast}C_{j}^{(I)}(E_{i}^{(I)}-E_{j}^{(I)})\mathbf{d}_{ij\nu}^{(I)}
   - \sum_{i}|C_{i}^{(I)}|^2 \sum_{\nu'}\frac{\mathcal{G}_{\nu',ij}^{(I)} - \mathcal{P}_{\nu'}^{(I)}}{M_{\nu'}} \cdot \nabla_{\nu'}(S_i^{(I)} - S_j^{(I)})
   \left[\sum_{j}|C_{j}^{(I)}|^2\nabla_\nu S_{j}^{(I)}-\nabla_\nu S_{i}^{(I)}\right]

The electronic equation of motion includes the coupled-trajectory decoherence K term:

.. math::

   \dot C^{(I)}_k(t) = -\frac{i}{\hbar}E^{(I)}_k(t)C^{(I)}_k(t)
   - \sum_j\sum_{\nu}{\bf d}^{(I)}_{kj\nu}(t)\cdot\dot{\bf R}^{(I)}_\nu(t)C^{(I)}_j(t)
   - \sum_{\nu}\frac{\mathcal{G}^{(I)}_{\nu,kj}(t)}{\hbar M_{\nu}}\cdot\left[\sum_{j}|C^{(I)}_{j}|^2\nabla_\nu S^{(I)}_{j}-\nabla_\nu S^{(I)}_{k}\right]C^{(I)}_{k}

where :math:`\mathcal{G}_{\nu,ij}` is the CRUNCH projected quantum momentum and :math:`\mathcal{P}_{\nu}` is the total quantum momentum.

Detailed descriptions of the CTv2 method are in :cite:`Agostini2016` and :cite:`Kim2022`.

.. note:: CTv2 requires multiple trajectories. The **molecules** and **istates** parameters must be lists.

+--------------------------------+------------------------------------------------+-----------------+
| Parameters                     | Work                                           | Default         |
+================================+================================================+=================+
| **molecules**                  | List of Molecule objects                       |                 |
| *(*:class:`Molecule`, *list)*  |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **thermostat**                 | Thermostat object                              | *None*          |
| (:class:`Thermostat`)          |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **istates**                    | List of initial states                         | *None*          |
| *(integer, list)*              |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **dt**                         | Time interval                                  | *0.5*           |
| *(double)*                     |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **nsteps**                     | Total step of nuclear propagation              | *1000*          |
| *(integer)*                    |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **nesteps**                    | Total step of electronic propagation           | *20*            |
| *(integer)*                    |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **elec_object**                | Electronic equation of motions                 | *'coefficient'* |
| *(string)*                     |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **propagator**                 | Electronic propagator                          | *'rk4'*         |
| *(string)*                     |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **l_print_dm**                 | Logical to print BO population and coherence   | *True*          |
| *(boolean)*                    |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **l_adj_nac**                  | Adjust nonadiabatic coupling to align phases   | *True*          |
| *(boolean)*                    |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **rho_threshold**              | Electronic density threshold for decoherence   | *0.01*          |
| *(double)*                     | term calculation                               |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **init_coefs**                 | Initial BO coefficients                        | *None*          |
| *(double/complex, 2D list)*    |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **l_crunch**                   | Use CRUNCH projected quantum momentum          | *True*          |
| *(boolean)*                    |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **l_dc_w_mom**                 | Use state-wise momentum for the phase term     | *True*          |
| *(boolean)*                    |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **l_traj_gaussian**            | Use trajectory-centered Gaussians for          | *False*         |
| *(boolean)*                    | nuclear density                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **t_cons**                     | Population conservation scheme                 | *2*             |
| *(integer)*                    |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **l_etot0**                    | Use constant total energy (at t=0) for         | *True*          |
| *(boolean)*                    | state-wise momentum                            |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **l_lap**                      | Include Laplacian ENC term                     | *False*         |
| *(boolean)*                    |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **l_real_pop**                 | Use |C_j|^2 for populations in quantum          | *True*          |
| *(boolean)*                    | momentum calculation                           |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **t_pc**                       | Phase correction scheme                        | *1*             |
| *(integer)*                    |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **l_en_cons**                  | Adjust momentum to enforce total energy        | *False*         |
| *(boolean)*                    | conservation                                   |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **artifact_expon**             | Exponent for width in nuclear density          | *0.2*           |
| *(double)*                     | estimation (trajectory Gaussians only)         |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **l_asymp**                    | Terminate dynamics in asymptotic region        | *False*         |
| *(boolean)*                    | (model systems only)                           |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **x_fin**                      | Asymptotic region boundary (a.u.)              | *25.0*          |
| *(double)*                     |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **use_gpu**                    | GPU acceleration mode                          | *False*         |
| *(boolean/string)*             |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **ncpus**                      | Number of CPUs for parallel QM calculations    | *1*             |
| *(integer)*                    |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **unit_dt**                    | Unit of time interval                          | *'fs'*          |
| *(string)*                     |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **out_freq**                   | Frequency of printing output                   | *1*             |
| *(integer)*                    |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+
| **verbosity**                  | Verbosity of output                            | *0*             |
| *(integer)*                    |                                                |                 |
+--------------------------------+------------------------------------------------+-----------------+


Detailed description of parameters
''''''''''''''''''''''''''''''''''''

- **molecules**

  This parameter defines molecular information for coupled trajectories.
  The data type must be a list of :ref:`Molecule <Objects Molecule>` objects.
  For example, if the number of coupled trajectories is 200, then **molecules** = *[mol1, mol2, ..., mol200]*
  where each element is a :ref:`Molecule <Objects Molecule>` object.

\

- **istates** *(integer, list)* - Default: *None*

  The BO coefficients and BO density matrices for coupled trajectories are initialized according to this parameter.
  The data type must be a list of integers with the same length as **molecules**.
  The possible range for each element is from *0* to ``molecule.nst - 1``.

\

- **dt** *(double)* - Default: *0.5*

  This parameter determines the time interval of the nuclear time steps.
  You can select the unit of time for the dynamics with the **unit_dt** parameter.

\

- **nsteps** *(integer)* - Default: *1000*

  This parameter determines the total number of nuclear time steps.

\

- **nesteps** *(integer)* - Default: *20*

  This parameter determines the number of electronic time steps between one nuclear time step
  for integration of the electronic equation of motion.

\

- **elec_object** *(string)* - Default: *'coefficient'*

  The **elec_object** parameter determines the representation for the electronic state.
  CTv2 only supports *'coefficient'*.

  + *'coefficient'*: Propagates the coefficients, i.e., :math:`\{C_{i}^{(I)}(t)\}`

\

- **propagator** *(string)* - Default: *'rk4'*

  This parameter determines the numerical integration method for the electronic equation of motion.
  Currently, only the RK4 algorithm (*'rk4'*) is available.

\

- **l_print_dm** *(boolean)* - Default: *True*

  This parameter determines whether to write output files for the density matrix elements ('BOPOP', 'BOCOH').

\

- **l_adj_nac** *(boolean)* - Default: *True*

  If this parameter is set to *True*, the signs of the NACVs are adjusted to match the phases to the previous time step.

\

- **rho_threshold** *(double)* - Default: *0.01*

  This parameter defines the numerical density threshold for the coherence.
  If the populations of two or more states are larger than this value, the electronic state is 'coherent'
  and the decoherence term is calculated.

\

- **init_coefs** *(double/complex, 2D list)* - Default: *None*

  This parameter defines the initial BO coefficients for all trajectories.
  The data type must be a 2D list where the outer list has length equal to the number of trajectories,
  and each inner list has length equal to ``molecule.nst``.
  If not given, the BO coefficients are initialized according to **istates**.

\

- **l_crunch** *(boolean)* - Default: *True*

  If set to *True*, the CRUNCH (coupled-trajectory using non-adiabatic coupling history) projected
  quantum momentum :math:`\mathcal{G}_{\nu,ij}` is used instead of the total quantum momentum
  :math:`\mathcal{P}_{\nu}` for the decoherence term. The projected quantum momentum is computed
  from state-pair-resolved Gaussian fitting, providing better resolution of the decoherence
  between specific pairs of states.

\

- **l_dc_w_mom** *(boolean)* - Default: *True*

  If set to *True*, the state-wise momentum :math:`\nabla_\nu S_i` is used for the phase term
  in the decoherence calculation. The state-wise momentum is computed from the BO energy and
  total energy conservation condition. If *False*, the phase is accumulated from BO forces over time.

\

- **l_traj_gaussian** *(boolean)* - Default: *False*

  If set to *True*, the nuclear density :math:`|\chi_i|^2` is represented as a sum of
  trajectory-centered Gaussians instead of a single Gaussian centered at the average position.
  This provides a more accurate density representation at the cost of O(ntrajs\ :sup:`2`) computations.

\

- **t_cons** *(integer)* - Default: *2*

  Determines the average population conservation scheme:

  + *0*: No conservation
  + *1*: Scaling (only valid when **l_lap** = *True*)
  + *2*: Shift — adds a correction term to maintain average population conservation

\

- **l_etot0** *(boolean)* - Default: *True*

  If set to *True*, the total energy at t=0 is used as a constant for the state-wise momentum calculation.
  This avoids drift in the state-wise momentum caused by small numerical energy fluctuations.

\

- **l_lap** *(boolean)* - Default: *False*

  If set to *True*, the Laplacian of the exact nuclear correction (ENC) term is included in the force calculation.

\

- **l_real_pop** *(boolean)* - Default: *True*

  If set to *True*, the electronic populations :math:`|C_j|^2` are used directly for
  :math:`|\chi_j|^2/|\chi|^2` in the quantum momentum calculation, rather than the Gaussian-fitted values.

\

- **t_pc** *(integer)* - Default: *1*

  Determines the phase correction scheme:

  + *0*: No phase correction
  + *1*: Use momentum P for phase correction
  + *2*: Use :math:`\sum_j \nabla S_j` for phase correction

\

- **l_en_cons** *(boolean)* - Default: *False*

  If set to *True*, the nuclear velocities are rescaled at every time step to enforce total energy conservation.

\

- **artifact_expon** *(double)* - Default: *0.2*

  Exponent used to scale the Gaussian width for nuclear density estimation.
  Only used when **l_traj_gaussian** = *True*.
  The sigma is scaled as :math:`\sigma \times 1.06 \times (\bar{\rho}_i \times N_{traj})^{-\text{artifact\_expon}}`.

\

- **l_asymp** *(boolean)* - Default: *False*

  If set to *True*, the dynamics is terminated when all trajectories reach the asymptotic region
  defined by **x_fin**. This option is intended for model systems only.

\

- **x_fin** *(double)* - Default: *25.0*

  Defines the asymptotic region boundary in atomic units. Used only when **l_asymp** = *True*.

\

- **use_gpu** *(boolean/string)* - Default: *False*

  Controls GPU acceleration for cross-trajectory calculations.
  See :ref:`GPU Acceleration` for details.

  + *False*: CPU mode (default)
  + *True*: Force GPU mode (requires PyTorch)
  + *'auto'*: Auto-detect GPU availability

\

- **ncpus** *(integer)* - Default: *1*

  Number of CPUs for parallel QM calculations across trajectories.
  When greater than 1, a multiprocessing pool distributes QM calculations.
  See :ref:`CPU Parallelization` for details.

\

- **unit_dt** *(string)* - Default: *'fs'*

  This parameter determines the unit of time for the simulation.

  + *'fs'*: femtosecond
  + *'au'*: atomic unit

\

- **out_freq** *(integer)* - Default: *1*

  PyUNIxMD prints and writes the dynamics information at every **out_freq** time steps.

\

- **verbosity** *(integer)* - Default: *0*

  This parameter determines the verbosity of the output files and stream.

  + **verbosity** :math:`\geq` *1*: Prints potential energy of all BO states and writes
    decoherence terms ('DOTPOPDEC').
  + **verbosity** :math:`\geq` *2*: Writes the NACVs ('NACV\_\ :math:`i`\_\ :math:`j`'),
    quantum momentum ('QMOM'), sigma ('SIGMA\_\ :math:`i`'), phase ('PHASE\_\ :math:`i`'),
    state-wise momentum ('MOM\_\ :math:`i`'), K matrix ('K\_\ :math:`i`\_\ :math:`j`'),
    and pseudo populations ('PSEUDOPOP').


Simulation Loop
''''''''''''''''''''''''''''''''''''

The CTv2 dynamics loop follows a three-phase architecture at each time step:

1. **Pre-QM phase** (serial per trajectory): Nuclear half-step position update, backup BO data
2. **QM phase** (parallelizable): QM electronic structure calculation for each trajectory.
   When ``ncpus > 1``, these are distributed across a multiprocessing pool.
3. **Post-QM phase** (serial per trajectory): NAC adjustment, velocity update, electronic propagation, coherence checks
4. **Cross-trajectory phase** (vectorized/GPU): Compute sigma, slope/center, quantum momentum, K matrix, population conservation

.. code-block:: text

   for istep in range(nsteps):
       # Pre-QM: half-step nuclear position update
       for itraj in range(ntrajs):
           calculate_force(itraj)
           cl_update_position(itraj)

       # QM: electronic structure (parallel if ncpus > 1)
       qm.get_data(...)  for each trajectory

       # Post-QM: velocity update + electronic propagation
       for itraj in range(ntrajs):
           calculate_force(itraj)
           cl_update_velocity(itraj)
           get_nacme() → el_run()
           check_decoherence(itraj)
           check_coherence(itraj)
           get_state_mom(itraj)
           get_phase(itraj)

       # Cross-trajectory: quantum momentum + decoherence
       calculate_sigma()
       calculate_slope()           # GPU-accelerable
       calculate_center()          # GPU-accelerable
       calculate_qmom()
       set_avg_pop_cons()

       # Output
       write_md_output(itraj) for each trajectory


Output Files
''''''''''''''''''''''''''''''''''''

CTv2 creates output under ``TRAJ_N/md/`` directories (one per trajectory):

+---------------------+-------------------------------------------------------+
| File                | Description                                           |
+=====================+=======================================================+
| MOVIE.xyz           | Trajectory with energies in comment lines             |
+---------------------+-------------------------------------------------------+
| FINAL.xyz           | Final geometry                                        |
+---------------------+-------------------------------------------------------+
| DENSITY             | Electronic state populations and coherences           |
+---------------------+-------------------------------------------------------+
| NACME               | Nonadiabatic coupling matrix elements                 |
+---------------------+-------------------------------------------------------+
| DOTPOPDEC           | Decoherence contributions (verbosity >= 1)            |
+---------------------+-------------------------------------------------------+
| PHASE\_\ *i*        | Phase term for state *i* (verbosity >= 2)             |
+---------------------+-------------------------------------------------------+
| MOM\_\ *i*          | State-wise momentum for state *i* (verbosity >= 2)    |
+---------------------+-------------------------------------------------------+
| K\_\ *i*\_\ *j*     | K matrix element (verbosity >= 2)                     |
+---------------------+-------------------------------------------------------+
| SIGMA\_\ *i*        | Gaussian width for state *i* (verbosity >= 2)         |
+---------------------+-------------------------------------------------------+
| PSEUDOPOP           | Pseudo populations (verbosity >= 2)                   |
+---------------------+-------------------------------------------------------+
| QMOM                | Quantum momentum (verbosity >= 2)                     |
+---------------------+-------------------------------------------------------+
| NACV\_\ *i*\_\ *j*  | Nonadiabatic coupling vectors (verbosity >= 2)        |
+---------------------+-------------------------------------------------------+


Example
''''''''''''''''''''''''''''''''''''

.. code-block:: python

   from molecule import Molecule
   import qm, mqc
   from misc import data
   import numpy as np

   ntraj = 200
   mass = 2000.
   data["X1"] = mass

   sigma = 2.0
   x0 = -25.
   k0 = 30.

   np.random.seed(1234)
   pos_list = np.random.normal(loc=x0, scale=sigma, size=ntraj)
   mom_list = np.random.normal(loc=k0, scale=0.5 / sigma, size=ntraj)

   mols = []
   istates = []
   for itraj in range(ntraj):
       geom = f"""
       1
       comment
       X1   {pos_list[itraj]}  {mom_list[itraj] / mass}
       """
       mol = Molecule(geometry=geom, ndim=1, nstates=2, ndof=1, unit_pos='au', l_model=True)
       mols.append(mol)
       istates.append(0)

   bo = qm.model.DAG(molecule=mols[0])

   md = mqc.CTv2(molecules=mols, istates=istates, dt=1., nsteps=1000, nesteps=20,
                  elec_object="coefficient", propagator="rk4", l_adj_nac=False,
                  rho_threshold=0.01, unit_dt="au", out_freq=10,
                  l_crunch=True, l_dc_w_mom=True, l_traj_gaussian=False,
                  t_cons=2, l_etot0=True, l_lap=False)

   md.run(qm=bo, output_dir="./")
