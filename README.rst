*****************************************************************
PyUNIxMD: Python-based UNIversal eXcited state Molecular Dynamics
*****************************************************************

.. image:: image/logo.svg
      :width: 600pt
      :align: center
      
PyUNIxMD is an object-oriented Python program for molecular dynamics simulations involving multiple electronic states.
It is mainly for studying the nonadiabatic dynamics of excited molecules.

Citation
========

When you publish a work using any part of PyUNIxMD code, please cite the following publication:

* PyUNIxMD: A Python-based excited state molecular dynamics package. *J. Comp. Chem.* **2021**, DOI: `10.1002/jcc.26711 <https://doi.org/10.1002/jcc.26711>`_

* Coupled- and independent-trajectory approaches based on the exact factorization using the PyUNIxMD package. *Top. Cur. Chem.* **2022**, DOI: `10.1007/s41061-021-00361-7 <https://doi.org/10.1007/s41061-021-00361-7>`_

Requirements
============
* Python 3.6 or later
* Numpy >= 1.20.0
* Scipy >= 1.6.0
* Cython https://cython.org
* BLAS/LAPACK libraries or Math Kernel Library

You can easily install the latest Numpy, Scipy and Cython via Python's pip command.

::

  $ pip install --upgrade numpy scipy Cython

GPU Acceleration (Optional)
===========================
PyUNIxMD supports GPU acceleration for coupled-trajectory methods (CTv2) to speed up
cross-trajectory calculations. This is particularly beneficial for large trajectory ensembles
(500+ trajectories).

**Supported Hardware:**

* Apple Silicon (M1/M2/M3/M4) via Metal Performance Shaders (MPS)
* NVIDIA GPUs via CUDA

**Installation:**

Install PyTorch to enable GPU acceleration:

::

  $ pip install torch

**Usage:**

GPU acceleration is disabled by default. To enable it, set ``use_gpu=True`` or
``use_gpu='auto'`` when creating the dynamics object.

.. code-block:: python

  import mqc

  # CPU mode (default)
  md = mqc.CTv2(molecules=mols, ...)

  # Force GPU mode
  md = mqc.CTv2(molecules=mols, use_gpu=True, ...)

  # Auto-detect GPU
  md = mqc.CTv2(molecules=mols, use_gpu='auto', ...)

You can also control the GPU mode via environment variable:

::

  $ export PYUNIXMD_USE_GPU=false   # Force CPU mode (default)
  $ export PYUNIXMD_USE_GPU=true    # Force GPU mode
  $ export PYUNIXMD_USE_GPU=auto    # Auto-detect

**Performance:**

Typical speedups on Apple Silicon for CTv2 cross-trajectory calculations:

* 500 trajectories: ~5x speedup
* 1000 trajectories: ~10x speedup
* 2000 trajectories: ~15-20x speedup

A benchmark script is provided at ``$PYUNIXMDHOME/tests/benchmark_gpu.py``.

Build
=====
You can build PyUNIxMD by the following command.

:: 

  $ python3 setup.py build_ext -b ./lib/

Examples
========
Without the aid of external QM programs, you can try the PyUNIxMD package with model systems.
The corresponding examples are:

* $PYUNIXMDHOME/examples/qm/SH-Shin_Metiu

* $PYUNIXMDHOME/examples/qm/SHXF-Shin_Metiu

* $PYUNIXMDHOME/examples/qm/CTv2-DAG

$PYUNIXMDHOME is the top-level directory where this file belongs.

In each directory, you can find the running script named run.py.

Before running example jobs, add the path of the PyUNIxMD package to your Python path.

::

  $ export PYTHONPATH=$PYUNIXMDHOME/src:$PYUNIXMDHOME:$PYTHONPATH

Then execute run.py as follows.

::

  $ python3 run.py >& log

Test
====
PyUNIxMD provides pytest-based tests in the $PYUNIXMDHOME/tests directory.
To run the tests, first set up the Python path and then use pytest.

::

  $ export PYTHONPATH=$PYUNIXMDHOME/src:$PYUNIXMDHOME:$PYTHONPATH
  $ cd $PYUNIXMDHOME/tests
  $ pytest -v

You can also run specific test categories using markers:

::

  $ pytest -m mqc -v      # Run all MQC tests
  $ pytest -m shxf -v     # Run only SHXF tests
  $ pytest -m bomd -v     # Run only BOMD tests

Utility scripts
===============
PyUNIxMD provides other Python scripts to analyze results of dynamics calculations.
To use the scripts, you need to add the path of the scripts.

::

  $ export PYTHONPATH=$PYUNIXMDHOME/util:$PYTHONPATH

Documentation
=============
If you have Sphinx, you can locally build the manual of PyUNIxMD by the following command.

::

  $ cd docs
  $ make html

