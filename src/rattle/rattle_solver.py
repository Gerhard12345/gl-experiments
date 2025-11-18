import numpy as np
from .rattle import (
    Hamiltonian,
    AlgebraicCondition,
    Rattle,
    Graph,
    ParametricSurface,
    ParameterManager,
)


class RollingSphereOnSurface:
    def __init__(self, surface_f, surface_df, position=np.array([0, 0, 0])):
        self.q_old = position
        hamiltonian = Hamiltonian(
            m=1.0,
            algebraic_condition=AlgebraicCondition(Graph(surface_f, surface_df)),
        )
        self.rattle = Rattle(
            hamiltonian=hamiltonian,
            p=np.array([0, 0, 0]),
            q=np.array(position),
            h=0.0005,
        )

    @property
    def energy(self):
        return self.rattle.hamiltonian(self.rattle.p, self.rattle.q)

    def step(self):
        self.rattle.step()
        self.dq = self.rattle.q - self.q_old
        self.q_old = self.rattle.q

    def get_rotation_axis(self):
        surface_dq = np.array(
            [
                [1, 0],
                [0, 1],
                self.rattle.hamiltonian.algebraic_condition.surface.dq(
                    self.rattle.q[0:2]
                ),
            ]
        )
        surface_normal = np.cross(surface_dq[:, 0], surface_dq[:, 1])
        moving_direction = self.dq
        rot_axis = np.cross(surface_normal, moving_direction)
        return rot_axis


class RollingSphereOnParametricSurface:
    def __init__(
        self,
        analytical_domain,
        parameter_manager,
        surface_f,
        surface_df,
        position=np.array([0, 0, 0]),
    ):
        self.q_old = position
        hamiltonian = Hamiltonian(
            m=1.0,
            algebraic_condition=AlgebraicCondition(
                ParametricSurface(
                    analytical_domain, parameter_manager, surface_f, surface_df
                )
            ),
        )
        self.rattle = Rattle(
            hamiltonian=hamiltonian,
            p=np.array([0, 0, 0]),
            q=np.array(position),
            h=0.005,
        )

    @property
    def energy(self):
        return self.rattle.hamiltonian(self.rattle.p, self.rattle.q)

    def step(self):
        self.rattle.step()
        self.dq = self.rattle.q - self.q_old
        self.q_old = self.rattle.q

    def get_rotation_axis(self):
        surface_dq = np.array(
            [
                [1, 0],
                [0, 1],
                self.rattle.hamiltonian.algebraic_condition.surface.dq(
                    self.rattle.q[0:2]
                ),
            ]
        )
        surface_normal = np.cross(surface_dq[:, 0], surface_dq[:, 1])
        moving_direction = self.dq
        rot_axis = np.cross(surface_normal, moving_direction)
        return rot_axis
