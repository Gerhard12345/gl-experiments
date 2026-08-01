from customgl import SphericalCoordianteSphere
from customgl import Material
import numpy as np
from rattle import RollingSphereOnSurface as RattleRollingSphereOnSurface
from surfaces import Surface, ParameterManager, ParametricSurface


class RollingSphereOnSurface(SphericalCoordianteSphere):
    def __init__(
        self,
        surface: Surface,
        position=np.array([0, 0, 0]),
        material=Material(),
        r=1.0,
    ):
        super().__init__(position, material, r)
        if type(surface) is ParametricSurface:
            p0 = position[0]
            p1 = position[1]

            position[0] = surface.parametric_domain.fx(p0, p1)
            position[1] = surface.parametric_domain.fy(p0, p1)
            pm = ParameterManager(uv=[p0, p1], q_old=position[0:2])
            surface.parameter_manager = pm
        self.rolling_sphere = RattleRollingSphereOnSurface(surface, position=position)
        self.translate(np.roll(position, -1))
        self.energy = self.rolling_sphere.energy

    def update(self):
        super().update()
        q_old = self.rolling_sphere.q_old
        self.translate(np.roll(-q_old, -1))
        for _ in range(10):
            self.rolling_sphere.step()
        rotation_axis = np.roll(self.rolling_sphere.get_rotation_axis(), -1)
        self.rotate_axis(1 / self.r * np.linalg.norm(self.rolling_sphere.rattle.q - q_old), rotation_axis)
        self.translate(np.roll(self.rolling_sphere.rattle.q, -1))
        self.energy = self.rolling_sphere.energy
