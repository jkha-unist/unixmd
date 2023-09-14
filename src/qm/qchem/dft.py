from __future__ import division
from qm.qchem.qchem import QChem
from misc import au_to_A, eV_to_au
import os, shutil, re, textwrap, subprocess
import numpy as np

class DFT(QChem):
    """ Class for DFT method of Q-Chem 5.2

        :param object molecule: Molecule object
        :param string basis_set: Basis set information
        :param integer memory: Allocatable memory in the calculation
        :param integer nthreads: Number of threads in the calculation
        :param string functional: Exchange-correlation functional 
        :param integer scf_max_iter: Maximum number of SCF iterations
        :param integer scf_wf_tol: Wave function convergence for SCF iterations
        :param integer cis_max_iter: Maximum number of CIS iterations
        :param integer cis_en_tol: Energy convergence for CIS iterations
        :param integer cpscf_max_iter: Maximum number of CP iterations
        :param integer cpscf_grad_tol: Gradient convergence of CP iterations
        :param string root_path: Path for Q-Chem root directory
        :param string version: Q-Chem version
    """
    def __init__(self, molecule, basis_set="sto-3g", memory=2000, nthreads=1, \
        functional="blyp", scf_max_iter=50, scf_wf_tol=8, cis_max_iter=30, cis_en_tol=6, \
        cpscf_max_iter=30, cpscf_grad_tol=6, root_path="./", version="5.2"):
        # Initialize Q-Chem common variables
        super(DFT, self).__init__(basis_set, memory, root_path, nthreads, version)

        self.functional = functional
        self.scf_max_iter = scf_max_iter
        self.nthreads = nthreads
        self.scf_wf_tol = scf_wf_tol
        self.cis_max_iter = cis_max_iter
        self.cis_en_tol = cis_en_tol
        self.cpscf_max_iter = cpscf_max_iter
        self.cpscf_grad_tol = cpscf_grad_tol

        # Q-Chem can provide NACs
        molecule.l_nacme = False
        self.re_calc = True

    def get_data(self, molecule, base_dir, bo_list, dt, istep, calc_force_only):
        """ Extract energy, gradient and nonadiabatic couplings from (TD)DFT method

            :param object molecule: Molecule object
            :param string base_dir: Base directory
            :param integer,list bo_list: List of BO states for BO calculation
            :param double dt: Time interval
            :param integer istep: Current MD step
            :param boolean calc_force_only: Logical to decide whether calculate force only
        """
        super().get_data(base_dir, calc_force_only)
        self.write_xyz(molecule)
        if (molecule.nT == 0):
            self.get_input(molecule, bo_list, calc_force_only)
            self.run_QM(base_dir, istep, bo_list)
            self.extract_QM(molecule, bo_list, calc_force_only)
        elif (molecule.nT > 0):
            self.get_input_ISC(molecule, bo_list, calc_force_only)
            self.run_QM_ISC(base_dir, istep, bo_list, calc_force_only)
            self.extract_QM_ISC(molecule, bo_list, calc_force_only)

        self.move_dir(base_dir)

    def get_input_ISC(self, molecule, bo_list, calc_force_only):
        """ Generate Q-Chem input files: qchem_soc.in, qchem_singlet.in, qchem_triplet.in

            :param object molecule: Molecule object
            :param integer istep: Current MD step
            :param integer,list bo_list: List of BO states for BO calculation
        """
        # Split bo_list into singlet and triplet lists
        singlet_list = []
        triplet_list = []
        for ist in bo_list:
            if (molecule.states[ist].mult == 1):
                singlet_list.append(molecule.states[ist].sub_ist)
            if (molecule.states[ist].mult == 3):
                triplet_list.append(molecule.states[ist].sub_ist)
        
        # Make Q-Chem input file for SOC: qchem_soc.in
        if (not calc_force_only and self.calc_coupling):
            input_soc = ""

            # Molecular information such as charge, geometry
            input_molecule = textwrap.dedent(f"""\
            $molecule
            {int(molecule.charge)}  1
            """)

            for iat in range(molecule.nat_qm):
                input_molecule += f"{molecule.symbols[iat]}"\
                    + "".join([f"{i:15.8f}" for i in molecule.pos[iat]]) + "\n"
            input_molecule += "$end\n\n"
            input_soc += input_molecule

            # Job control to calculate SOC
            cis_nroot = max(molecule.nS-1, molecule.nT)
            # Arguments about SCF, xc functional and basis set
            input_soc += textwrap.dedent(f"""\
            $rem
            JOBTYPE SP
            INPUT_BOHR TRUE
            METHOD {self.functional}
            BASIS {self.basis_set}
            SCF_CONVERGENCE {self.scf_wf_tol}
            SYMMETRY FALSE
            SYM_IGNORE TRUE
            """)
            # Arguments about TDDFT and SOC
            input_soc += textwrap.dedent(f"""\
            CIS_N_ROOTS {cis_nroot}
            CIS_TRIPLETS TRUE 
            CIS_TRIPLETS TRUE 
            CIS_CONVERGENCE {self.cis_en_tol}
            MAX_CIS_CYCLES {self.cis_max_iter}
            SET_ITER {self.cpscf_max_iter}
            SET_CONV {self.cpscf_grad_tol}
            CALC_SOC TRUE
            $end\n\n
            """)

            file_name = "qchem_soc.in"
            with open(file_name, "w") as f:
                f.write(input_soc)
        
        # Make Q-Chem input file for NAC within singlet: qchem_singlet.in
        input_singlet = textwrap.dedent(f"""\
        $molecule
        read
        $end\n
        """)
        # Job control for NAC for singlet states
        if (not calc_force_only and self.calc_coupling):
            # Arguments about SCF, xc functional and basis set
            input_singlet += textwrap.dedent(f"""\
            $rem
            JOBTYPE SP
            INPUT_BOHR TRUE
            METHOD {self.functional}
            BASIS {self.basis_set}
            SCF_GUESS READ
            SCF_CONVERGENCE {self.scf_wf_tol}
            SYMMETRY FALSE
            SYM_IGNORE TRUE
            """)

            # Arguments about TDDFT and NAC for singlet states
            input_singlet += textwrap.dedent(f"""\
            CIS_N_ROOTS {molecule.nS-1}
            CIS_SINGLETS TRUE
            CIS_TRIPLETS FALSE
            CIS_CONVERGENCE {self.cis_en_tol}
            CIS_GUESS_DISK TRUE
            CIS_GUESS_DISK_TYPE 2
            MAX_CIS_CYCLES {self.cis_max_iter}
            CALC_NAC TRUE
            CIS_DER_NUMSTATE {molecule.nS}
            SET_ITER {self.cpscf_max_iter}
            SET_CONV {self.cpscf_grad_tol}
            $end

            $derivative_coupling
            This is comment line
            """)

            for ist in range(molecule.nS):
                input_singlet += f"{ist}  "
            input_singlet += "\n$end\n\n"
        
        # Job control for force calculation
        input_force = ""
        guess = "READ"
        for ist in singlet_list:
            if (not calc_force_only and self.calc_coupling):
                input_force += textwrap.dedent(f"""\
                @@@

                $molecule
                read
                $end

                """)
            
            
            input_force += textwrap.dedent(f"""\
            $rem
            JOBTYPE force
            INPUT_BOHR TRUE
            METHOD {self.functional}
            BASIS {self.basis_set}
            SCF_GUESS {guess}
            SYMMETRY FALSE
            SYM_IGNORE TRUE
            """)
            
            # When ground state force is calculated, Q-Chem doesn't need CIS option.
            if (ist != 0):
                input_force += textwrap.dedent(f"""\
                CIS_N_ROOTS {molecule.nS-1}
                CIS_STATE_DERIV {ist}
                CIS_SINGLETS TRUE
                CIS_TRIPLETS FALSE
                CIS_CONVERGENCE {self.cis_en_tol}
                MAX_CIS_CYCLES {self.cis_max_iter}
                SET_ITER {self.cpscf_max_iter}
                SET_CONV {self.cpscf_grad_tol}
                CIS_GUESS_DISK TRUE
                CIS_GUESS_DISK_TYPE 2
                SKIP_CIS_RPA TRUE
                """)
            input_force += "$end\n\n"
        input_singlet += input_force

        file_name = "qchem_singlet.in"
        if ((not calc_force_only) or (calc_force_only and len(singlet_list) > 0)):
            with open(file_name, "w") as f:
                f.write(input_singlet)
        
        # Make Q-Chem input file for NAC within triplet: qchem_triplet.in
        input_triplet = textwrap.dedent(f"""\
        $molecule
        read
        $end\n
        """)
        # Job control for NAC for singlet states
        if (not calc_force_only and self.calc_coupling):
            # Arguments about SCF, xc functional and basis set
            input_triplet += textwrap.dedent(f"""\
            $rem
            JOBTYPE SP
            INPUT_BOHR TRUE
            METHOD {self.functional}
            BASIS {self.basis_set}
            SCF_GUESS READ
            SCF_CONVERGENCE {self.scf_wf_tol}
            SYMMETRY FALSE
            SYM_IGNORE TRUE
            """)

            # Arguments about TDDFT and NAC for triplet states
            input_triplet += textwrap.dedent(f"""\
            CIS_N_ROOTS {molecule.nT}
            CIS_SINGLETS FALSE
            CIS_TRIPLETS TRUE
            CIS_CONVERGENCE {self.cis_en_tol}
            CIS_GUESS_DISK TRUE
            CIS_GUESS_DISK_TYPE 0
            MAX_CIS_CYCLES {self.cis_max_iter}
            CALC_NAC TRUE
            CIS_DER_NUMSTATE {molecule.nT}
            SET_ITER {self.cpscf_max_iter}
            SET_CONV {self.cpscf_grad_tol}
            $end

            $derivative_coupling
            This is comment line
            """)

            for ist in range(molecule.nT):
                input_triplet += f"{ist+1}  "
            input_triplet += "\n$end\n\n"
        
        # Job control for force calculation
        input_force = ""
        guess = "READ"
        for ist in triplet_list:
            if (not calc_force_only and self.calc_coupling):
                input_force += textwrap.dedent(f"""\
                @@@

                $molecule
                read
                $end


                """)
            
            
            input_force += textwrap.dedent(f"""\
            $rem
            JOBTYPE force
            INPUT_BOHR TRUE
            METHOD {self.functional}
            BASIS {self.basis_set}
            SCF_GUESS {guess}
            SYMMETRY FALSE
            SYM_IGNORE TRUE
            """)
            
            input_force += textwrap.dedent(f"""\
            CIS_N_ROOTS {molecule.nT}
            CIS_STATE_DERIV {ist+1}
            CIS_SINGLETS FALSE
            CIS_TRIPLETS TRUE
            CIS_CONVERGENCE {self.cis_en_tol}
            MAX_CIS_CYCLES {self.cis_max_iter}
            SET_ITER {self.cpscf_max_iter}
            SET_CONV {self.cpscf_grad_tol}
            CIS_GUESS_DISK TRUE
            CIS_GUESS_DISK_TYPE 0
            SKIP_CIS_RPA TRUE
            """)
            input_force += "$end\n\n"
        input_triplet += input_force

        file_name = "qchem_triplet.in"
        if ((not calc_force_only) or (calc_force_only and len(triplet_list) > 0)):
            with open(file_name, "w") as f:
                f.write(input_triplet)


    def run_QM_ISC(self, base_dir, istep, bo_list, calc_force_only):
        """ Run (TD)DFT calculation and save the output files to qm_log directory

            :param string base_dir: Base directory
            :param integer istep: Current MD step
            :param integer,list bo_list: List of BO states for BO calculation
        """
        # Set environment variable 
        os.environ["QC"] = self.root_path
        path_qcenv = os.path.join(self.root_path, "qcenv.sh")
        command = f'env -i bash -c "source {path_qcenv} && env"'
        for line in subprocess.getoutput(command).split("\n"):
            key, value = line.split("=")
            os.environ[key] = value
        os.environ["QCSCRATCH"] = self.scr_qm_dir
        os.environ["QCLOCALSCR"] = self.scr_qm_dir

        # Run SOC calc
        if(not calc_force_only):
            qm_exec_command = f"$QC/bin/qchem -nt {self.nthreads} -save qchem_soc.in log_soc save_soc > qcprog_soc.info "
            os.system(qm_exec_command)
            os.system("cp -r save_soc save_singlet")
            os.system("cp -r save_soc save_triplet")
            os.system("cat log_soc > log")

        # Run singlet NAC (and force) calculation
        if ((not calc_force_only) or (calc_force_only and len(singlet_list) > 0)):
            qm_exec_command = f"$QC/bin/qchem -nt {self.nthreads} -save qchem_singlet.in log_singlet save_singlet >> qcprog_singlet.info "
            os.system(qm_exec_command)
            os.system("cat log_singlet >> log")

        # Run triplet NAC (and force) calculation
        if ((not calc_force_only) or (calc_force_only and len(triplet_list) > 0)):
            qm_exec_command = f"$QC/bin/qchem -nt {self.nthreads} -save qchem_triplet.in log_triplet save_triplet >> qcprog_triplet.info "
            os.system(qm_exec_command)
            os.system("cat log_triplet >> log")
        
        tmp_dir = os.path.join(base_dir, "qm_log")
        if (os.path.exists(tmp_dir)):
            log_step = f"log.{istep + 1}.{bo_list[0]}"
            shutil.copy("log", os.path.join(tmp_dir, log_step))
        
    def extract_QM_ISC(self, molecule, bo_list, calc_force_only):
        pass
    
    def get_input(self, molecule, bo_list, calc_force_only):
        """ Generate Q-Chem input files: qchem.in

            :param object molecule: Molecule object
            :param integer istep: Current MD step
            :param integer,list bo_list: List of BO states for BO calculation
        """
        # Make Q-Chem input file
        input_qc = ""

        # Molecular information such as charge, geometry
        input_molecule = textwrap.dedent(f"""\
        $molecule
        {int(molecule.charge)}  1
        """)

        for iat in range(molecule.nat_qm):
            input_molecule += f"{molecule.symbols[iat]}"\
                + "".join([f"{i:15.8f}" for i in molecule.pos[iat]]) + "\n"
        input_molecule += "$end\n\n"
        input_qc += input_molecule

        # Job control to calculate NAC
        if (not calc_force_only and self.calc_coupling):
            # Arguments about SCF, xc functional and basis set
            input_nac = textwrap.dedent(f"""\
            $rem
            JOBTYPE SP
            INPUT_BOHR TRUE
            METHOD {self.functional}
            BASIS {self.basis_set}
            SCF_CONVERGENCE {self.scf_wf_tol}
            SYMMETRY FALSE
            SYM_IGNORE TRUE
            """)

            # Arguments about TDDFT and NAC
            input_nac += textwrap.dedent(f"""\
            CIS_N_ROOTS {molecule.nst-1}
            CIS_TRIPLETS FALSE
            CIS_CONVERGENCE {self.cis_en_tol}
            MAX_CIS_CYCLES {self.cis_max_iter}
            CALC_NAC TRUE
            CIS_DER_NUMSTATE {molecule.nst}
            SET_ITER {self.cpscf_max_iter}
            SET_CONV {self.cpscf_grad_tol}
            $end

            $derivative_coupling
            This is comment line
            """)

            for ist in range(molecule.nst):
                input_nac += f"{ist}  "
            input_nac += "\n$end\n\n"
            input_qc += input_nac

        # Job control to calculate force
        input_force = ""

        # BOMD: calc_force_only = F, self_calc_coupling = F
        # In BOMD, read in molecule section, scf_guess and skip_scf are not valid
        guess = "SAD"; skip = "FALSE"
        for ist in bo_list:
            if (not calc_force_only and self.calc_coupling):
                guess = "READ"; skip = "TRUE"
                input_force = textwrap.dedent(f"""\
                @@@

                $molecule
                read
                $end

                """)

            input_force += textwrap.dedent(f"""\
            $rem
            JOBTYPE force
            INPUT_BOHR TRUE
            METHOD {self.functional}
            BASIS {self.basis_set}
            SCF_GUESS {guess}
            SYMMETRY FALSE
            SYM_IGNORE TRUE
            """)

            # When ground state force is calculated, Q-Chem doesn't need CIS option.
            if (ist != 0):
                input_force += textwrap.dedent(f"""\
                CIS_N_ROOTS {molecule.nst-1}
                CIS_STATE_DERIV {ist}
                CIS_TRIPLETS FALSE
                CIS_CONVERGENCE {self.cis_en_tol}
                MAX_CIS_CYCLES {self.cis_max_iter}
                SET_ITER {self.cpscf_max_iter}
                SET_CONV {self.cpscf_grad_tol}
                """)

                # CIS solution isn't saved in scratch.
                if (not calc_force_only and self.calc_coupling):
                    input_force += textwrap.dedent(f"""\
                    CIS_GUESS_DISK TRUE
                    CIS_GUESS_DISK_TYPE 2
                    SKIP_CIS_RPA TRUE
                    """)
            input_force += "$end\n\n"

            input_qc += input_force

        file_name = "qchem.in"
        with open(file_name, "w") as f:
            f.write(input_qc)

    def run_QM(self, base_dir, istep, bo_list):
        """ Run (TD)DFT calculation and save the output files to qm_log directory

            :param string base_dir: Base directory
            :param integer istep: Current MD step
            :param integer,list bo_list: List of BO states for BO calculation
        """
        # Set environment variable 
        os.environ["QC"] = self.root_path
        path_qcenv = os.path.join(self.root_path, "qcenv.sh")
        command = f'env -i bash -c "source {path_qcenv} && env"'
        for line in subprocess.getoutput(command).split("\n"):
            key, value = line.split("=")
            os.environ[key] = value
        os.environ["QCSCRATCH"] = self.scr_qm_dir
        os.environ["QCLOCALSCR"] = self.scr_qm_dir

        #TODO: MPI binary
        qm_exec_command = f"$QC/bin/qchem -nt {self.nthreads} qchem.in log save > qcprog.info "

        # Run Q-Chem
        os.system(qm_exec_command)

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
        file_name = "log"
        with open(file_name, "r") as f:
            log = f.read()

        if (not calc_force_only):
            # Ground state energy
            energy = re.findall('Total energy in the final basis set =\s*([-]*\S*)', log)
            energy = np.array(energy, dtype=np.float64)
            molecule.states[0].energy = energy[0]

            # Excited state energy
            if (molecule.nst > 1):
                energy = re.findall('Total energy for state\s*\d*.\s*([-]*\S*)', log)
                energy = np.array(energy, dtype=np.float64)

                for ist, en in enumerate(energy):
                    if ist < molecule.nst - 1:
                        molecule.states[ist + 1].energy = en

        # Adiabatic force 
        tmp_f = "Gradient of\D*\s*" 
        num_line = int(molecule.nat_qm / 6)
        if (num_line >= 1):
            tmp_f += ("\s*\d*\s*\d*\s*\d*\s*\d*\s*\d*\s*\d*"
                 + ("\s*\d?\s*" + "([-]*\S*)\s*" * 6) * 3) * num_line

        dnum = molecule.nat_qm % 6
        tmp_f += "\s*\d*" * dnum
        tmp_f += ("\s*\d?\s*" + "([-]*\S*)\s*" * dnum) * 3

        force = re.findall(tmp_f, log)
        force = np.array(force, dtype=np.float64)

        # Q-Chem provides energy gradient not force
        force = -force

        for index, ist in enumerate(bo_list):
            iline = 0; iiter = 0
            for iiter in range(num_line):
                tmp_force = np.transpose(force[index][18 * iiter:18 * (iiter + 1)].reshape(3, 6, order="C"))
                for iat in range(6):
                    molecule.states[ist].force[6 * iline + iat] = np.copy(tmp_force[iat])
                iline += 1

            if (dnum != 0):
                if (num_line != 0):
                    tmp_force = np.transpose(force[index][18 * (iiter + 1):].reshape(3, dnum, order="C"))
                else:
                    tmp_force = np.transpose(force[index][0:].reshape(3, dnum, order="C"))

                for iat in range(dnum):
                    molecule.states[ist].force[6 * iline + iat] = np.copy(tmp_force[iat])

        # NACs
        if (not calc_force_only and self.calc_coupling):
            tmp_nac = "with ETF[:]*\s*Atom\s*X\s*Y\s*Z\s*[-]*" + ("\s*\d*\s*" + "([-]*\S*)\s*"*3) * molecule.nat_qm
            nac = re.findall(tmp_nac, log)
            nac = np.array(nac, dtype=np.float64)

            kst = 0
            for ist in range(molecule.nst):
                for jst in range(ist + 1, molecule.nst):
                    molecule.nac[ist, jst] = nac[kst].reshape(molecule.nat_qm, 3, order='C')
                    molecule.nac[jst, ist] = - molecule.nac[ist, jst]
                    kst += 1


