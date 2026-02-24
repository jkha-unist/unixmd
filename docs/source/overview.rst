===========================
PyUNIxMD Overview
===========================

Features
---------------------------
The features of PyUNIxMD are as follows.

- Conventional (non)adiabatic dynamics

  - Born-Oppenheimer molecular dynamics (BOMD)
  - Ehrenfest dynamics :cite:`Ehrenfest1927`
  - Fewest switches surface hopping (FSSH) dynamics :cite:`Tully1990` with ad hoc decoherence corrections :cite:`Granucci2010`

.. Padding

- Decoherence based on exact factorization

  - Surface hopping based on exact factorization (SHXF) method :cite:`Ha2018`
  - Ehrenfest dynamics based on exact factorization (EhXF) method
  - Decoherence-induced surface hopping with exact factorization (SHXFv2 / DISH-XF) :cite:`Ha2018,Kim2022`
  - Coupled-trajectory mixed quantum-classical (CTMQC) method :cite:`Agostini2016`
  - Coupled-trajectory v2 (CTv2) with CRUNCH, state-wise momentum, and population conservation :cite:`Kim2022`

.. Padding

- Performance and scalability

  - GPU acceleration for cross-trajectory calculations (PyTorch with MPS/CUDA support)
  - CPU parallelization for QM calculations across trajectories (multiprocessing)
  - Vectorized NumPy operations with einsum for coupled-trajectory dynamics

.. Padding

- Accessible interface to external QM programs and built-in model Hamiltonians

  - COLUMBUS :cite:`Lischka2011`: SA-CASSCF
  - Molpro :cite:`Werner2012`: SA-CASSCF
  - Gaussian 09 :cite:`Frisch2009`: TDDFT
  - Q-Chem :cite:`qchem2015`: TDDFT
  - TURBOMOLE :cite:`Ahlrichs1989`: TDDFT
  - TeraChem :cite:`Ufimtsev2008_1,Ufimtsev2009_1,Ufimtsev2009_2`: SI-SA-REKS (SSR)
  - DFTB+ :cite:`Hourahine2020`: TDDFTB, DFTB/SSR
  - Model Hamiltonians: Tully :cite:`Tully1990`, Shin-Metiu :cite:`Shin1995`

.. Padding

- Accessible interface to deal with strong light-matter interaction within cavity quantum electrodynamics (cQED)

  - Jaynes-Cummings model :cite:`Jaynes1963`

.. Padding

- Numerical calculation of time-derivative nonadiabatic couplings (TDNACs) :cite:`Ryabinkin2015`
- QM/MM functionalities
- Utility scripts in Python

Authors
---------------------------
The current version of PyUNIxMD has been developed by Seung Kyu Min, In Seong Lee, Jong-Kwon Ha, Daeho Han, Kicheol Kim, Tae In Kim, Sung Wook Moon in the Theoretical/Computational Chemistry Group for Excited State Phenomena of Ulsan National Institute of Science and Technology (UNIST). 


Citation
---------------------------
Please cite the following work when publishing results from PyUNIxMD program:

\I. S. Lee, J.-K. Ha, D. Han, T. I. Kim, S. W. Moon, & S. K. Min. (2021). PyUNIxMD: A Python-based excited state molecular dynamics package. Journal of Computational Chemistry, 42:1755-1766. 2021

\T. I. Kim, J.-K. Ha, & S. K. Min. (2022). Coupled- and independent-trajectory approaches based on the exact factorization using the PyUNIxMD package. Topics in Current Chemistry, 380:153-179. 2022

..
  Acknowledgement
  ---------------------------
  This is acknowledgement.


Program Structure
---------------------------

For a detailed description of the code organization, core classes, MQC class hierarchy,
simulation loop, and build pipeline, see :ref:`Code Structure`.

