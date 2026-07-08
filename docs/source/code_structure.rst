.. _Code Structure:

===========================
Code Structure
===========================

This section provides an overview of the PyUNIxMD source code organization, core data structures,
class hierarchy, and the build pipeline.

Directory Layout
---------------------------

.. code-block:: text

   PyUNIxMD/
   ├── src/                         # Main source code
   │   ├── molecule.py              # Molecule & State classes
   │   ├── polariton.py             # QED polariton support
   │   ├── thermostat.py            # Thermostat implementations
   │   ├── trajectory.py            # Trajectory utilities
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
   │   │   ├── bomd.py, eh.py, sh.py, ct.py, shxf.py
   │   │   └── el_propagator C sources
   │   │
   │   ├── qm/                      # Quantum chemistry interfaces
   │   │   ├── qm_calculator.py     # Base QM calculator class
   │   │   ├── columbus/            # COLUMBUS (SA-CASSCF, MRCI)
   │   │   ├── molpro/              # Molpro (SA-CASSCF)
   │   │   ├── gaussian09/          # Gaussian 09 (TDDFT)
   │   │   ├── qchem/               # Q-Chem (TDDFT)
   │   │   ├── turbomole/           # TURBOMOLE (TDDFT)
   │   │   ├── terachem/            # TeraChem (SSR)
   │   │   ├── dftbplus/            # DFTB+ (TDDFTB, SSR)
   │   │   ├── gamess/              # GAMESS interface
   │   │   └── model/               # Built-in model Hamiltonians
   │   │       ├── sac.py           # Single avoided crossing
   │   │       ├── dac.py           # Double avoided crossing
   │   │       ├── dag.py           # Dual avoided + gap model
   │   │       ├── ecr.py           # Extended coupling with reflection
   │   │       ├── shin_metiu.py    # Shin-Metiu model
   │   │       └── file_io.py       # File-based input model
   │   │
   │   ├── mm/                      # Molecular mechanics interfaces
   │   │   ├── mm_calculator.py     # Base MM calculator class
   │   │   └── tinker.py            # Tinker interface
   │   │
   │   ├── qed/                     # QED model Hamiltonians
   │   │   ├── qed_calculator.py    # Base QED calculator class
   │   │   └── jaynes_cummings.py   # Jaynes-Cummings model
   │   │
   │   ├── cpa/                     # Classical path approximation
   │   │
   │   └── lib/                     # Cython/C source for electronic propagators
   │       ├── mqc/                 # Standard MQC propagators
   │       │   ├── el_propagator.pyx / .c
   │       │   ├── el_propagator_xf.pyx / .c        # SHXF propagator
   │       │   ├── el_propagator_xfv2.pyx / .c      # SHXFv2 propagator
   │       │   ├── el_propagator_ct.pyx / .c         # CT propagator
   │       │   └── el_propagator_ctv2.pyx / .c       # CTv2 propagator
   │       ├── mqc_qed/             # QED electronic propagators
   │       └── cioverlap/           # CI overlap calculations
   │
   ├── lib/                         # Compiled shared libraries (.so)
   ├── docs/                        # Sphinx documentation
   ├── tests/                       # Test suite (pytest)
   ├── examples/                    # Example input scripts
   ├── util/                        # Utility scripts
   └── setup.py                     # Build configuration


Core Classes
---------------------------

PyUNIxMD is an object-oriented program consisting of several key classes closely connected with each other:

- :class:`Molecule` defines a target system. A molecule object contains information of the electronic states as well as the geometry.
  To run cQED, :class:`Polariton` must be defined instead of :class:`Molecule`, which deals with the polaritonic states.

- :class:`MQC` has information about molecular dynamics. Each nonadiabatic dynamics method (Ehrenfest, surface hopping, etc.) comprises its subclasses.
  To run cQED, :class:`MQC_QED` must be defined instead of :class:`MQC`.

- :class:`QM_calculator` interfaces several QM programs (Molpro, Gaussian 09, DFTB+, etc.) and methodologies to perform electronic structure calculations.

- :class:`MM_calculator` enables QM/MM calculations using external softwares such as Tinker.

- :class:`QED_calculator` deals with strong light-matter interaction using built-in model Hamiltonians (Jaynes-Cummings model).

- :class:`Thermostat` controls temperature of a target system.

PyUNIxMD takes advantage of the inheritance feature to organize functionalities and simplify the codes by sharing the common parameters and methods.

For detailed information of each class, see :ref:`PyUNIxMD Objects <Objects>`.

Molecule and State
''''''''''''''''''''''

The :class:`Molecule` class (``src/molecule.py``) holds the complete system state:

- **Geometry**: ``pos`` (positions), ``vel`` (velocities), ``mass`` (atomic masses)
- **Electronic states**: ``states`` — a list of :class:`State` objects, each containing ``energy``, ``force``, ``coef`` (complex coefficient), and ``multiplicity``
- **Couplings**: ``nacme`` (scalar nonadiabatic coupling matrix elements), ``nac`` (nonadiabatic coupling vectors), ``socme`` (spin-orbit coupling)
- **Density matrix**: ``rho`` (complex density matrix)
- **Flag**: ``l_nacme`` — indicates whether the QM calculator provides scalar NACME directly (True) or NAC vectors that need conversion (False)


MQC Class Hierarchy
---------------------------

All MQC methods inherit from the :class:`MQC` base class which implements the **Template Method pattern**.
The base class provides the velocity-Verlet nuclear propagation loop, while subclasses override
``calculate_force()``, ``update_energy()``, and ``run()`` for method-specific logic.

.. code-block:: text

   MQC (base)
   ├── Single-trajectory methods
   │   ├── BOMD       — Born-Oppenheimer MD
   │   ├── Eh         — Ehrenfest dynamics
   │   ├── SH         — Fewest-switches surface hopping
   │   ├── SHXF       — SH + exact factorization decoherence
   │   ├── EhXF       — Ehrenfest + exact factorization decoherence
   │   └── SHXFv2     — Decoherence-induced SH with exact factorization (DISH-XF)
   │
   └── Multi-trajectory methods
       ├── CT         — Coupled-trajectory MQC (CTMQC)
       └── CTv2       — Coupled-trajectory v2 (vectorized, GPU/parallel support)


+----------------+----------------+-----------------------------+
| MQC Method     | Class Name     | Key Feature                 |
+================+================+=============================+
| BOMD           | ``BOMD``       | Single-state dynamics       |
+----------------+----------------+-----------------------------+
| Ehrenfest      | ``Eh``         | Mean-field electronic EOM   |
+----------------+----------------+-----------------------------+
| FSSH           | ``SH``         | Stochastic surface hopping  |
+----------------+----------------+-----------------------------+
| SHXF           | ``SHXF``       | XF decoherence + hopping    |
+----------------+----------------+-----------------------------+
| EhXF           | ``EhXF``       | XF decoherence + Ehrenfest  |
+----------------+----------------+-----------------------------+
| SHXFv2         | ``SHXFv2``     | DISH-XF with CRUNCH         |
+----------------+----------------+-----------------------------+
| CTMQC          | ``CT``         | Coupled trajectories         |
+----------------+----------------+-----------------------------+
| CTv2           | ``CTv2``       | Vectorized CT + GPU/parallel|
+----------------+----------------+-----------------------------+


Simulation Loop (Template Method)
-----------------------------------

All single-trajectory methods follow the velocity-Verlet pattern in ``run()``:

1. **Init**: Create ``md/`` directory, call ``qm.get_data()`` for initial geometry
2. **Main loop**:

   a. ``calculate_force()`` → ``cl_update_position()`` (half-step)
   b. ``qm.get_data()`` at new geometry (with ``backup_bo()``/``reset_bo()`` bracketing)
   c. ``adjust_nac()`` for phase alignment (if NAC vectors, not NACME)
   d. ``calculate_force()`` → ``cl_update_velocity()`` (half-step)
   e. ``mol.get_nacme()`` → ``el_run()`` (electronic propagation via Cython)
   f. Method-specific: hopping (SH), decoherence corrections, XF updates
   g. ``update_energy()`` → ``write_md_output()`` (controlled by ``out_freq``)
   h. Pickle restart checkpoint each step

Multi-trajectory methods (CT, CTv2) nest trajectory iteration inside the time-step loop,
adding cross-trajectory quantum momentum calculations after each time step.


Cython Build Pipeline
---------------------------

Performance-critical electronic propagation is implemented in C and wrapped via Cython:

1. Source files in ``src/lib/`` (``.pyx`` Cython wrappers + ``.c`` implementations)
2. Built via ``python3 setup.py build_ext -b ./lib/``
3. Produces shared objects in ``lib/``:

   - ``libmqc`` — Standard MQC electronic propagator
   - ``libmqcxf`` — SHXF electronic propagator
   - ``libmqcxfv2`` — SHXFv2 electronic propagator
   - ``libctmqc`` — CT electronic propagator
   - ``libctmqcv2`` — CTv2 electronic propagator
   - ``libcioverlap`` — CI overlap calculations
   - ``libmqc_qed``, ``libmqcxf_qed`` — QED electronic propagators

Each ``el_run()`` function serves as the entry point called from Python during the dynamics loop.

.. note:: Edit ``setup.py`` lines 24-29 to select the math library (LAPACK or MKL) and set library paths
   before building.


Tests and Examples
---------------------------

- **Tests** (``tests/``): Uses pytest with parameterized test cases covering all MQC methods.
  Test IDs follow the format ``TEST-{method}-{width_scheme}-{momentum_scheme}``.
  Reference data is stored in ``tests/reference/TEST-*/``.

- **Examples** (``examples/``): Running scripts organized by QM program (e.g., ``examples/qm/CTv2-DAG/run.py``).
