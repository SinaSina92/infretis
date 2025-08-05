import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # (only affects C++ backend, not Python logging)

import logging

# Suppress WARNINGs too — only show ERROR and CRITICAL
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("jax._src.dispatch").setLevel(logging.ERROR)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
# logging.basicConfig(level=logging.ERROR)

# If needed, suppress all DEBUG logs globally
# logging.basicConfig(level=logging.WARNING)

# Now import JAX and other libraries
import jax.numpy as jnp
from jax import grad, jit
import numpy as np

from ase.calculators.calculator import Calculator, all_changes


@jit
def acosbound(x):
    return jnp.arccos(jnp.clip(x, -1.0, 1.0))

@jit
def Vbond(x1, x2, k, r0):
    rij = x1 - x2
    rij_norm = jnp.linalg.norm(rij)
    V = 0.5 * k * (rij_norm - r0) ** 2
    return V

@jit
def Vangle(x1, x2, x3, k, theta0):
    rij = x1 - x2
    rkj = x3 - x2
    theta = acosbound(
        jnp.dot(rij, rkj) / (jnp.linalg.norm(rij) * jnp.linalg.norm(rkj))
    )
    V = 0.5 * k * (theta - theta0) ** 2
    return V


Fbond = jit(grad(Vbond, argnums=(0, 1)))
Fangle = jit(grad(Vangle, argnums=(0, 1, 2)))


class BondedInteractions(Calculator):
    """Non-periodic bonded interaction potential"""
    def __init__(
        self,
        bonds: list[tuple[float, float, int, int]],
        angles: list[tuple[float, float, int, int, int]],
        **kwargs,
        ):
        super().__init__(**kwargs)
        self.bonds = bonds
        self.angles = angles
        self.implemented_properties=["energy", "forces"]

    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes = all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = self.bonded_potential(atoms.positions)
        self.results["forces"] = self.bonded_forces(atoms.positions)


    def bonded_potential(self, pos):
        pot = 0.0

        for bond in self.bonds:
            pot += Vbond(pos[bond[2]], pos[bond[3]], bond[0], bond[1])

        for angle in self.angles:
            pot += Vangle(
                pos[angle[2]],
                pos[angle[3]],
                pos[angle[4]],
                angle[0],
                angle[1],
            )

        return pot

    def bonded_forces(self, pos):
        force = np.zeros(pos.shape)
        for bond in self.bonds:
            f1, f2 = Fbond(pos[bond[2]], pos[bond[3]], bond[0], bond[1])
            force[bond[2], :] += -f1
            force[bond[3], :] += -f2

        for angle in self.angles:
            f1, f2, f3 = Fangle(
                pos[angle[2]],
                pos[angle[3]],
                pos[angle[4]],
                angle[0],
                angle[1],
            )
            force[angle[2], :] += -f1
            force[angle[3], :] += -f2
            force[angle[4], :] += -f3

        return force
