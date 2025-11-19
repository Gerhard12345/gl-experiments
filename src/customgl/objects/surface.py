"""
A simple module for generating triangulated surfaces
"""

from collections import namedtuple
import numpy as np
from numpy.typing import NDArray

from .objects3d import Object3d
from .material import Material


AnalyticalDomain = namedtuple("AnalyticalDomain", ["fx", "fy", "Jf", "range_u", "range_v"])


class Mesh(Object3d):
    """
    The mesh class represents a flat linear triangulation
    """

    def __init__(
        self,
        analytical_domain: AnalyticalDomain,
        h_u: float,
        h_v: float,
        position: NDArray[np.float32],
        material: Material,
    ):
        super().__init__(position, material, scale=np.array([1, 1, 1]))
        self.analytical_domain = analytical_domain
        self.h_u = h_u
        self.h_v = h_v

    @property
    def v_values(self):
        """
        Returns the v parameter values
        """
        return np.linspace(*self.analytical_domain.range_v, int(1 / self.h_v + 1))

    @property
    def u_values(self):
        """
        Returns the u parameter values
        """
        return np.linspace(*self.analytical_domain.range_u, int(1 / self.h_u + 1))

    @property
    def parameter_grid(self):
        """
        Returns the parameter grid
        """
        return ((u, v) for u in self.u_values for v in self.v_values)

    @property
    def x_values(self):
        """
        Returns the x component of the mapped parameter grid values
        """
        return (self.analytical_domain.fx(u_val, v_val) for u_val, v_val in self.parameter_grid)

    @property
    def y_values(self):
        """
        Returns the y component of the mapped parameter grid values
        """
        return (self.analytical_domain.fy(u_val, v_val) for u_val, v_val in self.parameter_grid)

    @property
    def nodes(self):
        """
        Returns the nodes of the triangulation
        """
        return (
            [
                self.analytical_domain.fx(u_val, v_val),
                self.analytical_domain.fy(u_val, v_val),
            ]
            for u_val, v_val in self.parameter_grid
        )

    @property
    def linear_index(self):
        """
        Returns the linear_index of the nodes
        """
        return (y + x * len(self.v_values) for x in range(len(self.u_values) - 1) for y in range(len(self.v_values) - 1))

    @property
    def trigs(self):
        """
        Returns the trigs: [0,1,len(self.v_values]
        and [1,1+len(self.v_values),len(self.v_values]
        for each node. A node is represented by its linear index.
        """
        trigs = (
            ([base, base + 1, base + len(self.v_values)], [base + 1, base + 1 + len(self.v_values), base + len(self.v_values)])
            for base in self.linear_index
        )
        return (trig for trigtuple in trigs for trig in trigtuple)


class MeshedSurface(Mesh):
    """
    The Surface class extends the mesh class by a z component
    """

    def __init__(self, analytical_domain: AnalyticalDomain, f_z, d_f, h_u, h_v, position, material):
        super().__init__(analytical_domain, h_u, h_v, position, material)
        self.f_z = f_z
        self.df = d_f
        h_u = (analytical_domain.range_u[1] - analytical_domain.range_u[0]) / int(1 / h_u + 1)
        h_v = (analytical_domain.range_v[1] - analytical_domain.range_v[0]) / int(1 / h_v + 1)
        self.texture_coords_u = np.zeros([len(self.u_values), len(self.v_values)])
        self.texture_coords_v = np.zeros([len(self.u_values), len(self.v_values)])
        for i, (u, u_next) in enumerate(zip(self.u_values, self.u_values[1:])):
            for j, (v, v_next) in enumerate(zip(self.v_values, self.v_values[1:])):
                delta_f_v = np.array(self.value(u_next, v_next)) - np.array(self.value(u_next, v))
                delta_f_u = np.array(self.value(u_next, v_next)) - np.array(self.value(u, v_next))
                if j == 0:
                    self.texture_coords_v[i, j] = self.value(u_next, v_next)[1]
                else:
                    self.texture_coords_v[i, j] = self.texture_coords_v[i, j - 1] + delta_f_v[1]
                if i == 0:
                    self.texture_coords_u[i, j] = self.value(u_next, v_next)[0]
                else:
                    self.texture_coords_u[i, j] = self.texture_coords_u[i - 1, j] + delta_f_u[0]
                self.texture_coords_u[i, j] = self.value(u, v)[0]
                self.texture_coords_v[i, j] = self.value(u, v)[1]
        self.texture_coords_u += np.min(self.texture_coords_u)
        self.texture_coords_v += np.min(self.texture_coords_v)
        self.texture_coords_u = self.texture_coords_u.flatten()
        self.texture_coords_v = self.texture_coords_v.flatten()
        fullnodes = [
            (
                *np.roll(self.value(u, v), -1),
                *np.roll(self.normal(u, v), -1),
                texture_coord_u,
                texture_coord_v,
                *np.roll(self.tangent(u, v), -1),
                *np.roll(self.bitangent(u, v), -1),
            )
            for (u, v), texture_coord_u, texture_coord_v in zip(self.parameter_grid, self.texture_coords_u, self.texture_coords_v)
        ]
        vertices = [[fullnodes[trig[0]], fullnodes[trig[2]], fullnodes[trig[1]]] for trig in self.trigs]
        self._nvertices = len(vertices) * 3
        vertices = np.array(vertices).flatten()
        self._vertices = vertices.astype(np.float32)
        self._indices = np.array([range(0, self._nvertices)], dtype=np.uint32)
        self.translate(self.position)

    def value(self, u, v):
        return [self.analytical_domain.fx(u, v), self.analytical_domain.fy(u, v), self.f_z(u, v)]

    def normal(self, u, v):
        n = np.cross(self.tangent(u, v), self.bitangent(u, v))
        return n / np.linalg.norm(n)

    def tangent(self, u, v):
        t = np.array([self.analytical_domain.Jf(u, v)[0, 0], self.analytical_domain.Jf(u, v)[0, 1], self.df(u, v)[0]])
        return t / np.linalg.norm(t)

    def bitangent(self, u, v):
        b = np.array([self.analytical_domain.Jf(u, v)[1, 0], self.analytical_domain.Jf(u, v)[1, 1], self.df(u, v)[1]])
        return b / np.linalg.norm(b)

    @property
    def z_values(self):
        """
        Return the z values at the mesh nodes
        """
        return (self.f_z(u, v) for u, v in self.parameter_grid)


class MeshedSurfaceWithNormalOffset(MeshedSurface):
    """
    The Surface class extends the mesh class by a z component
    """

    def __init__(self, analytical_domain: AnalyticalDomain, f_z, d_f, h_u, h_v, position, material, r):
        self.r = r
        super().__init__(analytical_domain, f_z, d_f, h_u, h_v, position, material)

    def value(self, u, v):
        n = super().normal(u, v)
        n = n / np.linalg.norm(n)
        n *= self.r
        return [self.analytical_domain.fx(u, v) - n[0], self.analytical_domain.fy(u, v) - n[1], self.f_z(u, v) - n[2]]


circular_analytical_domain = AnalyticalDomain(
    lambda u, v: u * np.cos(v),
    lambda u, v: u * np.sin(v),
    lambda u, v: np.matrix([[np.cos(v), np.sin(v)], [-u * np.sin(v), u * np.cos(v)]]),
    [0.5, 2.5],
    [0, 2 * np.pi],
)

unit_square = AnalyticalDomain(
    lambda u, v: u,
    lambda u, v: v,
    lambda u, v: np.matrix([[1, 0], [0, 1]]),
    [-1, 1],
    [-1, 1],
)


class MeshedSurfaceWall(Object3d):
    """
    The Surface class extends the mesh class by a z component
    """

    def __init__(self, meshed_surface: MeshedSurface, material: Material, bottom_height: float):
        super().__init__(position=meshed_surface.position, scale=meshed_surface.scale, material=material)
        self.meshed_surface = meshed_surface
        self.texture_coords_u = np.zeros([len(self.meshed_surface.u_values), 2])
        self.texture_coords_v = np.zeros([len(self.meshed_surface.u_values), 2])
        vertices = []
        for v, v_next in zip(self.meshed_surface.v_values, self.meshed_surface.v_values[1:]):
            u_values = [self.meshed_surface.analytical_domain.range_u[0], self.meshed_surface.analytical_domain.range_u[1]]
            fullnodes = []
            for side_index, u0 in enumerate(u_values):
                for u_local in [0, 1]:
                    for v_local in [v, v_next]:
                        x, y, z0 = self.meshed_surface.value(u0, v_local)
                        z = z0 * u_local + (bottom_height) * (1 - u_local)
                        texture_coord_u = v_local
                        texture_coord_v = z
                        t = self.meshed_surface.bitangent(u0, v_local)
                        t[2] = 0
                        t /= np.linalg.norm(t)
                        b = np.array([0, 0, 1])
                        n = 2 * (side_index - 0.5) * np.cross(t, b)
                        fullnodes.append(
                            [*np.roll([x, y, z], -1), *np.roll(n, -1), texture_coord_u, texture_coord_v, *np.roll(t, -1), *np.roll(b, -1)]
                        )
            vertices.extend(fullnodes[0])
            vertices.extend(fullnodes[3])
            vertices.extend(fullnodes[1])
            vertices.extend(fullnodes[0])
            vertices.extend(fullnodes[2])
            vertices.extend(fullnodes[3])

            vertices.extend(fullnodes[4])
            vertices.extend(fullnodes[5])
            vertices.extend(fullnodes[7])
            vertices.extend(fullnodes[4])
            vertices.extend(fullnodes[7])
            vertices.extend(fullnodes[6])
        self._nvertices = len(vertices) // 14
        vertices = np.array(vertices).flatten()
        print(self._nvertices)
        self._vertices = vertices.astype(np.float32)
        self._indices = np.array([range(0, self._nvertices)], dtype=np.uint32)
        self.translate(self.position)

    def value(self, u, v):
        v0 = self.meshed_surface.analytical_domain.range_v[1]
        return [
            self.meshed_surface.analytical_domain.fx(u, v0),
            self.meshed_surface.analytical_domain.fy(u, v0),
            v,
        ]

    def normal(self, u, v):
        n = np.cross(self.tangent(u, v), self.bitangent(u, v))
        return n / np.linalg.norm(n)

    def tangent(self, u, v):
        t = np.array(
            [
                self.meshed_surface.analytical_domain.Jf(u, v)[0, 0],
                self.meshed_surface.analytical_domain.Jf(u, v)[0, 1],
                self.meshed_surface.df(u, v)[0],
            ]
        )
        return t / np.linalg.norm(t)

    def bitangent(self, u, v):
        b = np.array(
            [
                self.meshed_surface.analytical_domain.Jf(u, v)[1, 0],
                self.meshed_surface.analytical_domain.Jf(u, v)[1, 1],
                self.meshed_surface.df(u, v)[1],
            ]
        )
        return b / np.linalg.norm(b)
