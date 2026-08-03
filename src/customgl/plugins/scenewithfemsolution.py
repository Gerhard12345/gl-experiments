from customgl import Scene
import numpy as np
from customgl import MeshedSurfaceWithNormalOffset, MeshedSurfaceWall
from customgl import RoomDefinition, build_room
from customgl import Object3d

from customgl import (
    Material,
    WoodenCeiling,
    GoldFoil,
    MuddyConcrete,
    TerraCottaTiles,
    WhiteBricks,
    WornMetal,
)

from typing import List
from .rollingsphereonsurface import RollingSphereOnSurface
from surfaces import AnalyticalDomain, ParametricSurface
from pyfemsolver.solverlib.geometry import Line, Region, Geometry
from pyfemsolver.solverlib.space import H1Space
from pyfemsolver.solverlib.solving import solve_bvp, set_boundary_values
from pyfemsolver.visual.visual import show_grid_function
from pyfemsolver.solverlib.meshing import generate_mesh
from pyfemsolver.solverlib.geometry import Line, Region, Geometry, DoubleSlitGeometry, SingleSlitGeometryWith3Domains
from pyfemsolver.solverlib.coefficientfunction import VariableCoefficientFunction, ConstantCoefficientFunction
from pyfemsolver.solverlib.forms import BilinearForm, LinearForm
from pyfemsolver.solverlib.integrators import Laplace


def u_bnd(x: float, y: float) -> float:  # pylint:disable=C0116
    return (x - 0.5) ** 3 + (y - 0.5) ** 3


g = VariableCoefficientFunction({1: u_bnd, 2: u_bnd, 3: u_bnd, 4: u_bnd}, f_shape=(1, 1))

orders = [1, 4]


def generate_vertices():
    # singleslit = SingleSlitGeometryWith3Domains()
    # mesh = generate_mesh(singleslit, max_gradient=0.4)
    doubleslit = DoubleSlitGeometry()
    mesh = generate_mesh(doubleslit, max_gradient=0.75)

    coordinates_of_trigs = [[mesh.points[point].coordinates for point in trig.points] for trig in mesh.trigs]
    bary_centric = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    vertices = [
        [
            (*np.roll([float(lii[0]), float(lii[1]), 0.0], -1), *np.roll([0.0, 0.0, 1.0], -1), 0, 0, *bary, 0.0, 0.0, 0.0)
            for lii, bary in zip(li, bary_centric)
        ]
        for li in coordinates_of_trigs
    ]
    return vertices, mesh
    # space = H1Space(mesh, order, dirichlet_indices=[1, 2, 3, 4])

    # laplace = Laplace(ConstantCoefficientFunction(1), space, is_boundary=False)
    # bilinearform = BilinearForm([laplace])
    # linearform = LinearForm([])

    # set boundary values
    # u = space.create_gridfunction()
    # set_boundary_values(u, space, g)

    # solve_bvp(bilinearform, linearform, u, space)
    # ax, mini, maxi = show_grid_function(u, space, vrange=(-6.75, 0.25), n_subdivision=16)


class FemMesh(Object3d):
    def __init__(self, position=[0, 0, 0], scale=[1, 1, 1], material=Material()):
        super().__init__(position=position, scale=scale, material=material)
        vertices, mesh = generate_vertices()
        self._nvertices = len(vertices) * 3
        print(self._nvertices)
        self.cull_face = False
        vertices = np.array(vertices).flatten()
        self._vertices = vertices.astype(np.float32)
        self._indices = np.array([range(0, self._nvertices)], dtype=np.uint32)
        self.scale(scale).translate(position)


class SceneWithFemSolution(Scene):

    def __init__(self):
        super(SceneWithFemSolution, self).__init__()
        mesh = FemMesh()
        self.objects.append(mesh)
