import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms
from ase.md.verlet import VelocityVerlet
from ase.units import fs, kB
import matplotlib.pyplot as plt
import os


class Gaussian(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, v=0.5, c=0.5, **kwargs):
        super().__init__(**kwargs)
        self.v = v
        self.c = c

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()[:, 0]  # 1D x-coord

        energy = 0.0
        forces = np.zeros_like(atoms.get_positions())

        for i, x in enumerate(positions):
            exp_term = np.exp(-(x**2) / (2 * self.c**2))
            energy += self.v * exp_term
            forces[i, 0] = self.v * exp_term * x / (self.c**2)

        self.results['energy'] = energy
        self.results['forces'] = forces


class DoubleDip(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, v=0.5, s=0.3, d=1.0, **kwargs):
        super().__init__(**kwargs)
        self.v = v
        self.s = s
        self.d = d
        self.A = v / 5
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()[:, 0]

        energy = 0.0
        forces = np.zeros_like(atoms.get_positions())

        for i, x in enumerate(positions):
            # Energy terms
            exp_main = np.exp(-x**2 / (2 * self.s**2))
            xd_left = x + self.d
            xd_right = x - self.d
            exp_left = np.exp(-xd_left**2 / (2 * self.s**2))
            exp_right = np.exp(-xd_right**2 / (2 * self.s**2))

            # Energy
            e = self.v * exp_main + self.A * (exp_left + exp_right)
            energy += e

            # force terms
            f_main = self.v * x * exp_main / (self.s**2)
            f_left = self.A * xd_left * exp_left / (self.s**2)
            f_right = self.A * xd_right * exp_right / (self.s**2)

            forces[i, 0] = f_main + f_left + f_right

        self.results['energy'] = energy
        self.results['forces'] = forces
        
class FlatPotential(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, E0=0.0, **kwargs):
        super().__init__(**kwargs)
        self.E0 = E0

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n_atoms = len(atoms)
        self.results['energy'] = self.E0
        self.results['forces'] = np.zeros((n_atoms, 3))
