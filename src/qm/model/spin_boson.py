from __future__ import division
from qm.model.model import Model
import numpy as np

class Spin_Boson(Model):
    """ Class for one-mode spin-boson model BO calculation

        Two-state diabatic Hamiltonian of shifted parabolas with constant
        coupling (Agostini, Tavernelli, Ciccotti, Eur. Phys. J. B 91, 139
        (2018), Eqs. 37-38):
        V_+/- = A R^2 +/- B R + C, V_12 = V_21 = D

        :param object molecule: molecule object
        :param double A: harmonic curvature of the diabatic potentials
        :param double B: linear shift of the diabatic potentials
        :param double C: energy offset of the diabatic potentials
        :param double D: constant diabatic coupling
    """
    def __init__(self, molecule, A=1., B=3.5, C=3.0625, D=1.):
        # Initialize model common variables
        super(Spin_Boson, self).__init__(None)

        # Define parameters
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        # Set 'l_nacme' with respect to the computational method
        # Spin-Boson model can produce NACs, so we do not need to get NACME
        molecule.l_nacme = False

        # Spin-Boson model can compute the gradient of several states simultaneously
        self.re_calc = False

    def get_data(self, molecule, base_dir, bo_list, dt, istep, calc_force_only, traj=None):
        """ Extract energy, gradient and nonadiabatic couplings from one-mode
            spin-boson model BO calculation

            :param object molecule: molecule object
            :param string base_dir: base directory
            :param integer,list bo_list: list of BO states for BO calculation
            :param double dt: time interval
            :param integer istep: current MD step
            :param boolean calc_force_only: logical to decide whether calculate force only
            :param object traj: Trajectory object containing the calculator and trajectory
        """
        # Initialize diabatic Hamiltonian
        H = np.zeros((2, 2))
        dH = np.zeros((2, 2))
        unitary = np.zeros((2, 2))

        x = molecule.pos[0, 0]

        # Define Hamiltonian
        H[0, 0] = self.A * x ** 2 + self.B * x + self.C
        H[1, 1] = self.A * x ** 2 - self.B * x + self.C
        H[1, 0] = self.D
        H[0, 1] = H[1, 0]

        # Define a derivative of Hamiltonian
        dH[0, 0] = 2. * self.A * x + self.B
        dH[1, 1] = 2. * self.A * x - self.B
        dH[1, 0] = 0.
        dH[0, 1] = dH[1, 0]

        # Diagonalization
        a = 4. * H[1, 0] * H[0, 1] + (H[1, 1] - H[0, 0]) ** 2
        sqa = np.sqrt(a)
        tantheta = (H[1, 1] - H[0, 0] - sqa) / H[1, 0] * 0.5
        theta = np.arctan(tantheta)

        unitary[0, 0] = np.cos(theta)
        unitary[1, 0] = np.sin(theta)
        unitary[0, 1] = - np.sin(theta)
        unitary[1, 1] = np.cos(theta)

        # Extract adiabatic quantities
        molecule.states[0].energy = 0.5 * (H[0, 0] + H[1, 1]) - 0.5 * sqa
        molecule.states[1].energy = 0.5 * (H[0, 0] + H[1, 1]) + 0.5 * sqa

        molecule.states[0].force = - np.dot(unitary[:, 0], np.matmul(dH, unitary[:, 0]))
        molecule.states[1].force = - np.dot(unitary[:, 1], np.matmul(dH, unitary[:, 1]))

        molecule.nac[0, 1, 0, 0] = np.dot(unitary[:, 0], np.matmul(dH, unitary[:, 1])) / sqa
        molecule.nac[1, 0, 0, 0] = - molecule.nac[0, 1, 0, 0]
