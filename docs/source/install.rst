==============================
Installation
==============================

PyUNIxMD is a Python based program with a little C code for time-consuming parts
(e.g., electronic propagation or TDNACs) interfaced via Cython, therefore compilation is needed.

Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Python 3.6 (or later)

-  Numpy >= 1.20.0

-  Scipy >= 1.6.0

-  Cython https://cython.org

-  BLAS/LAPACK libraries or Math Kernel Library

If you don't have Numpy, Scipy or Cython, you can install them using :code:`pip` command.

.. code-block:: bash

   $ pip install --upgrade numpy scipy Cython


Compilation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can obtain PyUNIxMD code by putting following command.

.. code-block:: bash

   $ git clone https://github.com/skmin-lab/unixmd.git

You can compile the C routines by typing the following
command in the top-level directory of the program which contains setup.py file.

.. code-block:: bash

   $ cd unixmd/
   $ python3 setup.py build_ext -b ./lib/

You can select the type of math libraries and the path of math libraries by modifying **math_lib_type** and **math_lib_dir**
defined in the setup.py file as follows:

.. code-block:: python
   :linenos:

   from distutils.core import setup
   from distutils.extension import Extension
   from Cython.Distutils import build_ext

   import numpy as np

   # Selects the type of math libraries to be used; Available options: lapack, mkl
   math_lib_type = "lapack"

   # Directories including the math libraries
   math_lib_dir = "/my_disk/my_name/lapack/"

   ...

After successful compilation, you will need to add the source directory to your Python path,
where :code:`$PYUNIXMDHOME` is an environment variable for the top-level directory.

.. code-block:: bash

   $ export PYTHONPATH=$PYUNIXMDHOME/src:$PYUNIXMDHOME:$PYTHONPATH


GPU Acceleration (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyUNIxMD supports GPU acceleration for coupled-trajectory methods (CTv2) to speed up
cross-trajectory calculations. This is particularly beneficial for large trajectory ensembles
(500+ trajectories).

**Supported Hardware:**

- Apple Silicon (M1/M2/M3/M4) via Metal Performance Shaders (MPS)
- NVIDIA GPUs via CUDA

**Installation:**

Install PyTorch to enable GPU acceleration:

.. code-block:: bash

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

.. code-block:: bash

   $ export PYUNIXMD_USE_GPU=false   # Force CPU mode (default)
   $ export PYUNIXMD_USE_GPU=true    # Force GPU mode
   $ export PYUNIXMD_USE_GPU=auto    # Auto-detect

See :ref:`GPU Acceleration` for more details.


Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Without the aid of external QM programs, you can try the PyUNIxMD package with model systems.
The corresponding examples are:

- ``$PYUNIXMDHOME/examples/qm/SH-Shin_Metiu``
- ``$PYUNIXMDHOME/examples/qm/SHXF-Shin_Metiu``
- ``$PYUNIXMDHOME/examples/qm/CTv2-DAG``

In each directory, you can find the running script named ``run.py``.

Before running example jobs, add the path of the PyUNIxMD package to your Python path.

.. code-block:: bash

   $ export PYTHONPATH=$PYUNIXMDHOME/src:$PYUNIXMDHOME:$PYTHONPATH

Then execute ``run.py`` as follows.

.. code-block:: bash

   $ python3 run.py >& log


Tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyUNIxMD provides pytest-based tests in the ``$PYUNIXMDHOME/tests`` directory.
To run the tests, first set up the Python path and then use pytest.

.. code-block:: bash

   $ export PYTHONPATH=$PYUNIXMDHOME/src:$PYUNIXMDHOME:$PYTHONPATH
   $ cd $PYUNIXMDHOME/tests
   $ pytest -v

You can also run specific test categories using markers:

.. code-block:: bash

   $ pytest -m mqc -v      # Run all MQC tests
   $ pytest -m shxf -v     # Run only SHXF tests
   $ pytest -m bomd -v     # Run only BOMD tests


Utility Scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyUNIxMD provides other Python scripts to analyze results of dynamics calculations.
To use the scripts, you need to add the path of the scripts.

.. code-block:: bash

   $ export PYTHONPATH=$PYUNIXMDHOME/util:$PYTHONPATH


Building Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have Sphinx, you can locally build the manual of PyUNIxMD by the following command.

.. code-block:: bash

   $ cd docs
   $ make html
