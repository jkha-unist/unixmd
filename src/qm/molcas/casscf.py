from __future__ import division
from qm.molcas.molcas import Molcas
from misc import call_name
import os, shutil, re, textwrap
import numpy as np

class CASSCF(Molcas):
    """ Class for SA-CASSCF method of OpenMolcas (RASSCF + MCLR + ALASKA)

        :param object molecule: Molecule object
        :param string basis_set: Basis set information
        :param string memory: Allocatable memory in the calculations (in MB)
        :param string guess: Initial guess for (SA-)CASSCF method
        :param string guess_file: File containing initial guesses for (SA-)CASSCF calculations
        :param integer scf_max_iter: Maximum number of HF iterations
        :param double scf_en_tol: Energy convergence threshold for HF iterations
        :param integer mcscf_max_iter: Maximum number of (SA-)CASSCF macro iterations
        :param double mcscf_en_tol: Energy convergence threshold for (SA-)CASSCF iterations
        :param double mcscf_rot_tol: Orbital rotation convergence threshold for (SA-)CASSCF iterations
        :param double mcscf_grad_tol: Energy gradient convergence threshold for (SA-)CASSCF iterations
        :param integer active_elec: Number of electrons in active space
        :param integer active_orb: Number of orbitals in active space
        :param double cpscf_grad_tol: Convergence threshold for CP-MCSCF (MCLR) equations
        :param boolean l_nocsf: Neglect the CSF contribution to NACs (ALASKA NOCSF option)
        :param string qm_path: Path for the directory containing the 'pymolcas' driver
        :param integer nthreads: Number of threads in the calculations
        :param string version: Version of OpenMolcas
    """
    def __init__(self, molecule, basis_set="sto-3g", memory="2000", \
        guess="hf", guess_file="./molcas.RasOrb", scf_max_iter=100, scf_en_tol=1E-9, \
        mcscf_max_iter=200, mcscf_en_tol=1E-8, mcscf_rot_tol=1E-4, mcscf_grad_tol=1E-4, \
        active_elec=2, active_orb=2, cpscf_grad_tol=1E-7, l_nocsf=False, \
        qm_path="/usr/local/bin", nthreads=1, version="26.06"):
        # Initialize OpenMolcas common variables
        super(CASSCF, self).__init__(basis_set, memory, qm_path, nthreads, version)

        # Initialize OpenMolcas CASSCF variables
        # Set initial guess for CASSCF calculation
        self.guess = guess.lower()
        self.guess_file = guess_file
        if not (self.guess in ["hf", "read"]):
            error_message = "Invalid initial guess for CASSCF!"
            error_vars = f"guess = {self.guess}"
            raise ValueError (f"( {self.qm_method}.{call_name()} ) {error_message} ( {error_vars} )")

        # HF calculation for initial guess of CASSCF calculation
        self.scf_max_iter = scf_max_iter
        self.scf_en_tol = scf_en_tol

        # CASSCF calculation
        self.mcscf_max_iter = mcscf_max_iter
        self.mcscf_en_tol = mcscf_en_tol
        self.mcscf_rot_tol = mcscf_rot_tol
        self.mcscf_grad_tol = mcscf_grad_tol
        self.active_elec = active_elec
        self.active_orb = active_orb
        self.cpscf_grad_tol = cpscf_grad_tol

        # ALASKA prints the total derivative coupling <Psi_j|d/dR Psi_i> including the
        # CSF contribution; NOCSF drops that term (translationally invariant form)
        self.l_nocsf = l_nocsf

        # Check the closed shell for systems
        if (not int(molecule.nelec) % 2 == 0):
            error_message = "Only closed shell configuration supported, check charge!"
            error_vars = f"Molecule.nelec = {int(molecule.nelec)}"
            raise ValueError (f"( {self.qm_method}.{call_name()} ) {error_message} ( {error_vars} )")

        # CASSCF calculation do not provide parallel computation
        # If your system provide parallel casscf, then this part should be removed
        if (self.nthreads > 1):
            error_message = "Parallel CASSCF not implemented!"
            error_vars = f"nthreads = {self.nthreads}"
            raise ValueError (f"( {self.qm_method}.{call_name()} ) {error_message} ( {error_vars} )")

        # Calculate number of inactive (doubly occupied) orbitals in CASSCF method
        if ((int(molecule.nelec) - self.active_elec) % 2 != 0):
            error_message = "Wrong number of active electrons for closed-shell inactive space!"
            error_vars = f"Molecule.nelec = {int(molecule.nelec)}, active_elec = {self.active_elec}"
            raise ValueError (f"( {self.qm_method}.{call_name()} ) {error_message} ( {error_vars} )")
        self.inactive_orb = int((int(molecule.nelec) - self.active_elec) / 2)

        # Set 'l_nacme' with respect to the computational method
        # CASSCF can produce NACs, so we do not need to get NACME from CIoverlap
        # CASSCF can compute the gradient of several states simultaneously,
        #        but self.re_calc is set to be true to reduce cost.
        molecule.l_nacme = False
        self.re_calc = True

    def get_data(self, molecule, base_dir, bo_list, dt, istep, calc_force_only, traj=None):
        """ Extract energy, gradient and nonadiabatic couplings from (SA-)CASSCF method

            :param object molecule: Molecule object
            :param string base_dir: Base directory
            :param integer,list bo_list: List of BO states for BO calculation
            :param double dt: Time interval
            :param integer istep: Current MD step
            :param boolean calc_force_only: Logical to decide whether calculate force only
            :param object traj: Trajectory object containing the calculator and trajectory
        """
        if (not calc_force_only):
            self.copy_files(istep)
        super().get_data(base_dir, calc_force_only)
        self.write_xyz(molecule)
        self.get_input(molecule, istep, bo_list, calc_force_only)
        self.run_QM(base_dir, istep, bo_list)
        self.extract_QM(molecule, bo_list, calc_force_only)
        self.move_dir(base_dir)

    def copy_files(self, istep):
        """ Copy necessary scratch files in previous step

            :param integer istep: Current MD step
        """
        # Copy required files to read initial guess
        # RASSCF writes the SA natural orbitals to 'molcas.RasOrb' in scr_qm
        if (self.guess == "read" and istep >= 0):
            # After T = 0.0 s
            shutil.copy(os.path.join(self.scr_qm_dir, "molcas.RasOrb"), \
                os.path.join(self.scr_qm_dir, "../molcas.RasOrb"))

    def get_input(self, molecule, istep, bo_list, calc_force_only):
        """ Generate OpenMolcas input files: molcas.input

            :param object molecule: Molecule object
            :param integer istep: Current MD step
            :param integer,list bo_list: List of BO states for BO calculation
            :param boolean calc_force_only: Logical to decide whether calculate force only
        """
        if (self.calc_coupling and molecule.nst < 2):
            error_message = "NACs require at least two states!"
            error_vars = f"Molecule.nst = {molecule.nst}"
            raise ValueError (f"( {self.qm_method}.{call_name()} ) {error_message} ( {error_vars} )")

        # Decide the orbital guess source for this step
        if (calc_force_only):
            # Reuse the orbitals written by the energy run at the same geometry
            shutil.copy("molcas.RasOrb", "guess.RasOrb")
            hf = False
            read_orb = True
        elif (self.guess == "read"):
            if (istep == -1):
                if (os.path.isfile(self.guess_file)):
                    # Copy guess file to current directory
                    shutil.copy(self.guess_file, "guess.RasOrb")
                    hf = False
                    read_orb = True
                else:
                    hf = True
                    read_orb = False
            elif (istep >= 0):
                # Move previous file to current directory
                os.rename("../molcas.RasOrb", "./guess.RasOrb")
                hf = False
                read_orb = True
        elif (self.guess == "hf"):
            hf = True
            read_orb = False

        # Make 'molcas.input' file
        input_molcas = ""

        # Geometry Block: 'geometry.xyz' is written in angstrom
        input_geom = textwrap.dedent(f"""\
        &GATEWAY
          Coord = geometry.xyz
          Basis = {self.basis_set}
          Group = NoSym
          NoCD

        &SEWARD

        """)
        input_molcas += input_geom

        # HF Block: initial guess option
        if (hf):
            input_hf = textwrap.dedent(f"""\
            &SCF
              Iterations = {self.scf_max_iter}
              Thresholds = {self.scf_en_tol} 1.0e-4 1.5e-4 0.2e-2

            """)
            input_molcas += input_hf

        # CASSCF Block: calculate energy option
        input_casscf = textwrap.dedent(f"""\
        &RASSCF
          Spin = 1
          Symmetry = 1
          nActEl = {self.active_elec} 0 0
          Inactive = {self.inactive_orb}
          Ras2 = {self.active_orb}
          CIRoot = {molecule.nst} {molecule.nst} 1
          Iter = {self.mcscf_max_iter} 100
          Thrs = {self.mcscf_en_tol} {self.mcscf_rot_tol} {self.mcscf_grad_tol}
        """)
        if (read_orb):
            input_casscf += "  FILEORB = $CurrDir/guess.RasOrb\n"
        input_molcas += input_casscf + "\n"

        # CASSCF Block: calculate gradient option (explicit MCLR for threshold control)
        for ist in bo_list:
            input_casscf_grad = textwrap.dedent(f"""\
            &MCLR
              THREshold = {self.cpscf_grad_tol}
              SALA = {ist + 1}

            &ALASKA
              ROOT = {ist + 1}

            """)
            input_molcas += input_casscf_grad

        # CASSCF Block: calculate NAC option
        if (not calc_force_only and self.calc_coupling):
            for ist in range(molecule.nst):
                for jst in range(ist + 1, molecule.nst):
                    input_casscf_nac = textwrap.dedent(f"""\
                    &MCLR
                      THREshold = {self.cpscf_grad_tol}
                      NAC = {ist + 1} {jst + 1}

                    &ALASKA
                      NAC = {ist + 1} {jst + 1}
                    """)
                    if (self.l_nocsf):
                        input_casscf_nac += "  NOCSF\n"
                    input_molcas += input_casscf_nac + "\n"

        # Write 'molcas.input' file
        file_name = "molcas.input"
        with open(file_name, "w") as f:
            f.write(input_molcas)

    def run_QM(self, base_dir, istep, bo_list):
        """ Run (SA-)CASSCF calculation and save the output files to qm_log directory

            :param string base_dir: Base directory
            :param integer istep: Current MD step
            :param integer,list bo_list: List of BO states for BO calculation
        """
        # Run OpenMolcas method
        qm_command = os.path.join(self.qm_path, "pymolcas")
        # OpenMP setting
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MOLCAS_NPROCS"] = "1"
        os.environ["MOLCAS_MEM"] = self.memory
        # Relative MOLCAS_WORKDIR puts the scratch inside scr_qm; wiped every step
        os.environ["MOLCAS_WORKDIR"] = "scr"
        os.environ["MOLCAS_PRINT"] = "NORMAL"
        command = f"{qm_command} -b 1 molcas.input > log 2>&1"
        os.system(command)
        # Copy the output file to 'qm_log' directory
        tmp_dir = os.path.join(base_dir, "qm_log")
        if (os.path.exists(tmp_dir)):
            log_step = f"log.{istep + 1}.{bo_list[0]}"
            shutil.copy("log", os.path.join(tmp_dir, log_step))

    def extract_QM(self, molecule, bo_list, calc_force_only):
        """ Read the output files to get BO information

            :param object molecule: Molecule object
            :param integer,list bo_list: List of BO states for BO calculation
            :param boolean calc_force_only: Logical to decide whether calculate force only
        """
        # Read 'log' file
        file_name = "log"
        with open(file_name, "r") as f:
            log_out = f.read()

        if (log_out.find("Happy landing") < 0):
            error_message = "OpenMolcas calculation failed, check the log file in scr_qm!"
            error_vars = f"log = {os.path.join(self.scr_qm_dir, 'log')}"
            raise Exception (f"( {self.qm_method}.{call_name()} ) {error_message} ( {error_vars} )")

        # Energy
        if (not calc_force_only):
            tmp_e = r'RASSCF root number\s+\d+\s+Total energy:\s+([-]?\d+\.\d+)'
            energy = re.findall(tmp_e, log_out)[:molecule.nst]
            energy = np.array(energy, dtype=np.float64)
            for ist in range(molecule.nst):
                molecule.states[ist].energy = energy[ist]

        # Per-atom row of the ALASKA gradient/NAC tables
        tmp_row = r'\s*\S+\s+([-]?\d\.\d+E[+-]\d+)\s+([-]?\d\.\d+E[+-]\d+)\s+([-]?\d\.\d+E[+-]\d+)\n' * molecule.nat_qm

        # Force: 'Molecular gradients' tables appear in bo_list order
        tmp_f = r'Molecular gradients[\s\S]*?X\s+Y\s+Z\s*\n\s*[-]+\s*\n' + tmp_row
        grad = re.findall(tmp_f, log_out)
        for kst, ist in enumerate(bo_list):
            force = np.array(grad[kst], dtype=np.float64)
            force = force.reshape(molecule.nat_qm, 3, order='C')
            molecule.states[ist].force = - np.copy(force)

        # NAC: for 'NAC = i j' ALASKA prints the total derivative coupling <Psi_j|d/dR Psi_i>
        if (not calc_force_only and self.calc_coupling):
            tmp_c = r'Total derivative coupling[\s\S]*?X\s+Y\s+Z\s*\n\s*[-]+\s*\n' + tmp_row
            nac_blocks = re.findall(tmp_c, log_out)
            kst = 0
            for ist in range(molecule.nst):
                for jst in range(ist + 1, molecule.nst):
                    nac = np.array(nac_blocks[kst], dtype=np.float64)
                    nac = nac.reshape(molecule.nat_qm, 3, order='C')
                    molecule.nac[jst, ist] = nac
                    molecule.nac[ist, jst] = - nac
                    kst += 1
