from __future__ import division
from qm.qm_calculator import QM_calculator
from misc import call_name
import os

class Molcas(QM_calculator):
    """ Class for common parts of OpenMolcas

        :param string basis_set: Basis set information
        :param string memory: Allocatable memory in the calculations (in MB, passed to MOLCAS_MEM)
        :param string qm_path: Path for the directory containing the 'pymolcas' driver
        :param integer nthreads: Number of threads in the calculations
        :param string version: Version of OpenMolcas
    """
    def __init__(self, basis_set, memory, qm_path, nthreads, version):
        # Save name of QM calculator and its method
        super().__init__()

        # Initialize OpenMolcas common variables
        self.basis_set = basis_set
        self.memory = memory

        self.qm_path = qm_path
        if (not os.path.isfile(os.path.join(self.qm_path, "pymolcas"))):
            error_message = "pymolcas driver not found in qm_path!"
            error_vars = f"qm_path = {self.qm_path}"
            raise FileNotFoundError (f"( {self.qm_method}.{call_name()} ) {error_message} ( {error_vars} )")

        self.nthreads = nthreads
        self.version = version

        if (isinstance(self.version, str)):
            # Output formats used in extract_QM are verified against OpenMolcas v26.x
            if (not self.version.startswith("26")):
                error_message = "Other versions not implemented!"
                error_vars = f"version = {self.version}"
                raise ValueError (f"( {self.qm_method}.{call_name()} ) {error_message} ( {error_vars} )")
        else:
            error_message = "Type of version must be string!"
            error_vars = f"version = {self.version}"
            raise TypeError (f"( {self.qm_method}.{call_name()} ) {error_message} ( {error_vars} )")
