from .objects3d import SphericalCoordianteSphere
from .material import Material
import numpy as np
from rattle.rattle_solver import RollingSphereOnSurface, RollingSphereOnParametricSurface
from rattle.rattle import ParameterManager


class RollingSphere(SphericalCoordianteSphere):
    def __init__(
        self,
        surface_f,
        surface_df,
        position=np.array([0, 0, 0]),
        material=Material(),
        r=1.0,
    ):
        super().__init__(position, material, r)
        self.rolling_sphere = RollingSphereOnSurface(surface_f, surface_df, position=position)
        self.translate(np.roll(position, -1))
        self.energy = self.rolling_sphere.energy

    def update(self):
        super().update()
        for _ in range(100):
            q_old = self.rolling_sphere.q_old
            self.translate(np.roll(-q_old, -1))
            self.rolling_sphere.step()
            rotation_axis = np.roll(self.rolling_sphere.get_rotation_axis(), -1)
            self.rotate_axis(1 / self.r * np.linalg.norm(self.rolling_sphere.rattle.q - q_old), rotation_axis)
            self.translate(np.roll(self.rolling_sphere.rattle.q, -1))
        self.energy = self.rolling_sphere.energy


class RollingSphereParametric(SphericalCoordianteSphere):
    def __init__(
        self,
        analytical_domain,
        surface_f,
        surface_df,
        position=np.array([0, 0, 0]),
        initial_parameter=np.array([0, 0]),
        material=Material(),
        r=1.0,
    ):
        super().__init__(position, material, r)
        pm = ParameterManager(uv=initial_parameter)
        self.rolling_sphere = RollingSphereOnParametricSurface(analytical_domain, pm, surface_f, surface_df, position=position)
        self.translate(np.roll(position, -1))
        self.energy = self.rolling_sphere.energy

    def update(self):
        super().update()
        q_old0 = self.rolling_sphere.q_old
        self.translate(np.roll(-q_old0, -1))
        for _ in range(10):
            self.rolling_sphere.step()
        rotation_axis = np.roll(self.rolling_sphere.get_rotation_axis(), -1)
        self.rotate_axis(1 / self.r * np.linalg.norm(self.rolling_sphere.rattle.q - q_old0), rotation_axis)
        self.translate(np.roll(self.rolling_sphere.rattle.q, -1))
        self.energy = self.rolling_sphere.energy
