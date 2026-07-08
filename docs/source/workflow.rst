==========================
Workflow
==========================

Molecular Dynamics
---------------------------

Here, we explain how to run MD calculations with PyUNIxMD.

You will make a running script for the MD calculation you want to perform. In your running script, you will create PyUNIxMD objects successively.
A typical template of the running script is the following:

.. code-block:: python
   :linenos:

   from molecule import Molecule
   import qm, mqc
   import thermostat

   geom = """
   <number of atoms>
   <comment>
   <symbol> <X> <Y> <Z> <V_X> <V_Y> <V_Z>
   <symbol> <X> <Y> <Z> <V_X> <V_Y> <V_Z>
   ...
   <symbol> <X> <Y> <Z> <V_X> <V_Y> <V_Z>
   """

   mol = Molecule(geometry=geom, nstates=NSTATES)

   qm = qm.QM_PROG.QM_METHOD(molecule=mol, ARGUMENTS)

   bathT = thermostat.THERMO_TYPE(ARGUMENTS)

   md = mqc.MD_TYPE(molecule=mol, thermostat=bathT, ARGUMENTS)

   md.run(qm=qm, ARGUMENTS)

**Line 1-3** import the PyUNIxMD packages for the below jobs.

**Line 5-14** set a target system you are interested in.
You need to prepare a string as an argument to specify initial geometry and velocities in extended XYZ format.
NSTATES means the number of adiabatic states considered in the dynamics calculations.
See :ref:`Molecule <Objects Molecule>` for the list of parameters.

.. note:: The ``mol`` object must be created first because it will be used for making other objects.

**Line 16** determines an electronic structure calculation program and its method to obtain QM information
such as energies, forces, and nonadiabatic coupling vectors. QM_PROG is the directory name where the QM interface
package is. QM_METHOD is a name of Python class specifying one of QM methods provided with that interface package.
See :ref:`QM_calculator <Objects QM_calculator>` for the list.

**Line 18** sets a thermostat. THERMO_TYPE is a name of Python class specifying how to control temperature. See :ref:`Thermostat <Objects Thermostat>` for the list.

**Line 20** determines a dynamics method you want to use. MD_TYPE is a name of Python class specifying one of MQC methods (BOMD, Eh, SH, SHXF, EhXF, SHXFv2, CT, CTv2). See :ref:`MQC <Objects MQC>` for the details.

**Line 22** runs the dynamics calculation.

Finally, you will execute your running script.

.. code-block:: bash

   $ python3 running_script.py

Running MD calculations with PyUNIxMD, you will obtain output files under the ``md/`` directory.
``qm_log/`` and ``mm_log/`` have logs of QM and MM calculations, respectively
(these directories are optional). ``RESTART.bin`` is a binary used to restart a dynamics calculation. See :ref:`MQC <Objects MQC>` for the details.

.. note:: Since default of **l_print_dm** is *True*, PyUNIxMD provides ``DENSITY`` regardless of **elec_object**.
   If **elec_object** is *"coefficient"* and you set **l_print_dm** to *False*, then ``DENSITY`` is not written.

Details of the MD output files and their formats are the following.

- MOVIE.xyz

This file contains positions, velocities, and energy information at each MD step (a trajectory).
Energy keywords are included in the comment line of each frame in the extended XYZ format.

.. code-block:: bash

   <number of atoms>
   step=<N> Ekin=<kinetic energy> Epot=<potential energy> Etot=<total MD energy> E0=<energy of state 0> E1=<energy of state 1> ...
   <symbol> <X> <Y> <Z> <V_X> <V_Y> <V_Z>
   <symbol> <X> <Y> <Z> <V_X> <V_Y> <V_Z>
   ...
   <number of atoms>
   step=<N+1> Ekin=<kinetic energy> Epot=<potential energy> Etot=<total MD energy> E0=<energy of state 0> E1=<energy of state 1> ...
   <symbol> <X> <Y> <Z> <V_X> <V_Y> <V_Z>
   <symbol> <X> <Y> <Z> <V_X> <V_Y> <V_Z>
   ...

- FINAL.xyz

This file contains the final position and velocity of an MD calculation,
with the same energy keyword format in the comment line as ``MOVIE.xyz``.

.. code-block:: bash

   <number of atoms>
   step=<last step> Ekin=<kinetic energy> Epot=<potential energy> Etot=<total MD energy> E0=<energy of state 0> ...
   <symbol> <X> <Y> <Z> <V_X> <V_Y> <V_Z>
   <symbol> <X> <Y> <Z> <V_X> <V_Y> <V_Z>
   ...

- DENSITY

This file shows the adiabatic populations (diagonal elements of the density matrix) and
off-diagonal coherences at each MD step.
Populations are listed first, followed by the upper-triangular off-diagonal elements
with real and imaginary parts written alternately.

.. code-block:: bash

   <MD step> <pop 0> <pop 1> ... <pop last> <Re(0,1)> <Im(0,1)> <Re(0,2)> <Im(0,2)> ... <Re(last-1,last)> <Im(last-1,last)>
   <MD step> <pop 0> <pop 1> ... <pop last> <Re(0,1)> <Im(0,1)> <Re(0,2)> <Im(0,2)> ... <Re(last-1,last)> <Im(last-1,last)>
   ...

- NACME

This file shows nonadiabatic coupling matrix elements at each MD step. Only the upper triangular portions are given because of antihermiticity.

.. code-block:: bash

   <MD step> <element 0, 1> <element 0, 2> ... <element last-1, last>
   <MD step> <element 0, 1> <element 0, 2> ... <element last-1, last>
   ...

- SHPROB

This file shows hopping probabilities from the running state to the others at each MD step.

.. code-block:: bash

   <MD step> <P(running -> 0)> <P(running -> 1)> ... <P(running -> last)>
   <MD step> <P(running -> 0)> <P(running -> 1)> ... <P(running -> last)>
   ...

- SHSTATE

This file shows the running state at each MD step.

.. code-block:: bash

   <MD step> <running>
   <MD step> <running>
   ...


Output Files by MQC Method
'''''''''''''''''''''''''''''''

The following table summarizes which output files are produced by each MQC method
at the default verbosity (``verbosity=0``).
Multi-trajectory methods (CT, CTv2) write per-trajectory output under ``TRAJ_N/md/``.

+----------------+------------+------------+----------+-------+---------+---------+
| File           | BOMD       | Eh         | SH       | SHXF  | EhXF    | SHXFv2  |
+================+============+============+==========+=======+=========+=========+
| MOVIE.xyz      | \+         | \+         | \+       | \+    | \+      | \+      |
+----------------+------------+------------+----------+-------+---------+---------+
| FINAL.xyz      | \+         | \+         | \+       | \+    | \+      | \+      |
+----------------+------------+------------+----------+-------+---------+---------+
| DENSITY        |            | \+         | \+       | \+    | \+      | \+      |
+----------------+------------+------------+----------+-------+---------+---------+
| NACME          |            | \+         | \+       | \+    | \+      | \+      |
+----------------+------------+------------+----------+-------+---------+---------+
| SHSTATE        |            |            | \+       | \+    | \+      | \+      |
+----------------+------------+------------+----------+-------+---------+---------+
| SHPROB         |            |            | \+       | \+    | \+      | \+      |
+----------------+------------+------------+----------+-------+---------+---------+

+----------------+------------+------------+
| File           | CT         | CTv2       |
+================+============+============+
| MOVIE.xyz      | \+         | \+         |
+----------------+------------+------------+
| FINAL.xyz      | \+         | \+         |
+----------------+------------+------------+
| DENSITY        | \+         | \+         |
+----------------+------------+------------+
| NACME          | \+         | \+         |
+----------------+------------+------------+

For a quick test of PyUNIxMD, see :ref:`Quick Start <Quick Start>` . Also, you can refer to scripts and log files in '$PYUNIXMDHOME/examples/' directory for practical calculations.


Polariton Dynamics
---------------------------

Similarly, you will make a running script for polariton dynamics. In your running script, you will create PyUNIxMD objects successively.
A typical template of the running script is the following:

.. code-block:: python
   :linenos:

   from polariton import Polariton
   import qm, qed, mqc_qed
   import thermostat

   geom = """
   <number of atoms>
   <comment>
   <symbol> <X> <Y> <Z> <V_X> <V_Y> <V_Z>
   <symbol> <X> <Y> <Z> <V_X> <V_Y> <V_Z>
   ...
   <symbol> <X> <Y> <Z> <V_X> <V_Y> <V_Z>
   """

   pol = Polariton(geometry=geom, nstates=NSTATES)

   qm = qm.QM_PROG.QM_METHOD(molecule=pol, ARGUMENTS)

   qed = qed.QED_METHOD(polariton=pol, ARGUMENTS)

   bathT = thermostat.THERMO_TYPE(ARGUMENTS)

   md = mqc_qed.MD_TYPE(polariton=pol, thermostat=bathT, ARGUMENTS)

   md.run(qed=qed, qm=qm, ARGUMENTS)

**Line 1-3** import the PyUNIxMD packages for the below jobs.

**Line 5-14** set a target system you are interested in.
You need to prepare a string as an argument to specify initial geometry and velocities in extended XYZ format.
NSTATES means the number of polaritonic states considered in the dynamics calculations.
See :ref:`Polariton <Objects Polariton>` for the list of parameters.

.. note:: The ``pol`` object must be created first because it will be used for making other objects.

**Line 16** determines an electronic structure calculation program and its method to obtain QM information
such as energies, forces, and nonadiabatic coupling vectors. QM_PROG is the directory name where the QM interface
package is. QM_METHOD is a name of Python class specifying one of QM methods provided with that interface package.
See :ref:`QM_calculator <Objects QM_calculator>` for the list.

**Line 18** determines a method for quantum electrodynamics calculation. QED_METHOD is a name of Python class
specifying one of QED methods privoded with that interface.
See :ref:`QED_calculator <Objects QED_calculator>` for the list.

**Line 20** sets a thermostat. THERMO_TYPE is a name of Python class specifying how to control temperature. See :ref:`Thermostat <Objects Thermostat>` for the list.

**Line 22** determines a dynamics method you want to use. MD_TYPE is a name of Python class specifying one of MQC_QED methods (BOMD, Eh, SH, SHXF, CT). See :ref:`MQC_QED <Objects MQC_QED>` for the details.

**Line 24** runs the dynamics calculation.

Finally, you will execute your running script.

.. code-block:: bash

   $ python3 running_script.py

After the polariton dynamics is finished, you will obtain similar file trees as above.
'md/' collects MD outputs, and 'qm_log/', 'mm_log/', and 'qed_log/' have logs of QM, MM, and QED calculations, respectively
(The latter three directories are optional). 'RESTART.bin' is a binary used to restart a dynamics calculation.
See :ref:`MQC_QED <Objects MQC_QED>` for the details.

Instead of BO-related output files (DENSITY, NACME), several QED-related output files (QEDPOPA, QEDCOHA, QEDPOPD, QEDCOHD, PNACME) will be generated.
The same output files (MOVIE.xyz, FINAL.xyz, SHPROB, SHSTATE) are produced for polariton dynamics.
In addition, polariton dynamics writes a separate ``MDENERGY`` file for energy information.

- MDENERGY

This file shows MD energies and energies of polaritonic states.

.. code-block:: bash

   <MD step> <kinetic energy> <potential energy> <total MD energy> <polaritonic energy 0> <polaritonic energy 1> ...
   <MD step> <kinetic energy> <potential energy> <total MD energy> <polaritonic energy 0> <polaritonic energy 1> ...
   ...

- QEDPOPA, QEDPOPD

These files show the polaritonic (with suffix 'A') and uncoupled (with suffix 'D') populations
(diagonal elements of the density matrix) at each MD step.

.. code-block:: bash

   <MD step> <population of state 0> <population of state 1> ... <population of last state>
   <MD step> <population of state 0> <population of state 1> ... <population of last state>
   ...

- QEDCOHA, QEDCOHD

These files show off-diagonal elements of the corresponding density matrix at each MD step. Only the upper triangular portions are given because of hermiticity. The real and imaginary part of each element are written alternately.

.. code-block:: bash

   <MD step> <Re. element 0, 1> <Im. element 0, 1> <Re. element 0, 2> <Im. element 0, 2> ... <Re. element last-1, last> <Im. element last-1, last>
   <MD step> <Re. element 0, 1> <Im. element 0, 1> <Re. element 0, 2> <Im. element 0, 2> ... <Re. element last-1, last> <Im. element last-1, last>
   ...

- PNACME

This file shows nonadiabatic coupling matrix elements between the polaritonic states at each MD step. Only the upper triangular portions are given because of antihermiticity.

.. code-block:: bash

   <MD step> <element 0, 1> <element 0, 2> ... <element last-1, last>
   <MD step> <element 0, 1> <element 0, 2> ... <element last-1, last>
   ...


Output Files by MQC_QED Method
'''''''''''''''''''''''''''''''''

The following table summarizes which output files are produced by each MQC_QED method
at the default verbosity (``verbosity=0``).
The multi-trajectory method (CT) writes per-trajectory output under ``TRAJ_N/md/``.

.. note:: QED coefficient files (QEDCOEFA, QEDCOEFD) are always written when ``elec_object="coefficient"``.
   Population and coherence files (QEDPOPA, QEDCOHA, QEDPOPD, QEDCOHD) require ``l_print_dm=True`` (default).

+----------------+--------+------+------+------+------+
| File           | BOMD   | Eh   | SH   | SHXF | CT   |
+================+========+======+======+======+======+
| MOVIE.xyz      | \+     | \+   | \+   | \+   | \+   |
+----------------+--------+------+------+------+------+
| FINAL.xyz      | \+     | \+   | \+   | \+   | \+   |
+----------------+--------+------+------+------+------+
| MDENERGY       | \+     | \+   | \+   | \+   | \+   |
+----------------+--------+------+------+------+------+
| QEDCOEFA       |        | \+   | \+   | \+   | \+   |
+----------------+--------+------+------+------+------+
| QEDCOEFD       |        | \+   | \+   | \+   | \+   |
+----------------+--------+------+------+------+------+
| QEDPOPA        |        | \+   | \+   | \+   | \+   |
+----------------+--------+------+------+------+------+
| QEDCOHA        |        | \+   | \+   | \+   | \+   |
+----------------+--------+------+------+------+------+
| QEDPOPD        |        | \+   | \+   | \+   | \+   |
+----------------+--------+------+------+------+------+
| QEDCOHD        |        | \+   | \+   | \+   | \+   |
+----------------+--------+------+------+------+------+
| PNACME         |        | \+   | \+   | \+   | \+   |
+----------------+--------+------+------+------+------+
| SHSTATE        |        |      | \+   | \+   |      |
+----------------+--------+------+------+------+------+
| SHPROB         |        |      | \+   | \+   |      |
+----------------+--------+------+------+------+------+

For a quick test for polariton dynamics, it will be added later.


