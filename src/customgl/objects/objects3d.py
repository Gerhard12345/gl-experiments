import numpy as np
from numpy.typing import NDArray
from .material import Material
from .transformations import Transformations
from typing import List


class Object3d:
    def __init__(self, position: NDArray[np.float32], material: Material, scale: NDArray[np.float32]):
        self.position = position
        self.scalev = scale
        self.material = material
        self._vertices = None
        self._indices: NDArray = None
        self.dynamics = np.matrix(np.identity(4))
        self.modelmat = np.matrix(np.identity(4))

    def get_n_trigs(self):
        return len(self._vertices) // 3

    def get_vertices(self) -> NDArray[np.float32]:
        return self._vertices

    def get_n_vertices(self) -> NDArray[np.float32]:
        return self._indices.size

    def get_indices(self) -> NDArray[np.float32]:
        return self._indices

    def scale(self, scale_xyz: NDArray[np.float32]):
        self.modelmat *= Transformations.scalemat(scale_xyz)
        return self

    def local_rot_x(self, angle: float):
        self.modelmat *= Transformations.localrotationmat_axis(self.position, angle, np.array([1.0, 0, 0]))
        return self

    def local_rot_y(self, angle: float):
        self.modelmat *= Transformations.localrotationmat_axis(self.position, angle, np.array([0, 1.0, 0]))
        return self

    def local_rot_z(self, angle: float):
        self.modelmat *= Transformations.localrotationmat_axis(self.position, angle, np.array([0, 0, 1.0]))
        return self

    def rotate_x(self, angle: float):
        self.rotate_axis(angle, np.array([1.0, 0.0, 0.0]))
        return self

    def rotate_y(self, angle: float):
        self.rotate_axis(angle, np.array([0.0, 1.0, 0.0]))
        return self

    def rotate_z(self, angle: float):
        self.rotate_axis(angle, np.array([0.0, 0.0, 1.0]))
        return self

    def rotate_axis(self, angle: float, axis: NDArray[np.float32]):
        self.modelmat *= Transformations.rotationmat_axis(angle, axis)
        return self

    def translate(self, position: NDArray[np.float32]):
        self.modelmat *= Transformations.translationmat(position)
        return self

    def applydynamics(self):
        self.modelmat *= self.dynamics

    def add_dynamic(self, dynamic) -> None:
        self.dynamics *= dynamic

    def update(self) -> None:
        self.applydynamics()


class Cube(Object3d):
    def __init__(self, position: NDArray[np.float32], material: Material, scale: NDArray[np.float32]):
        super().__init__(position=position, material=material, scale=scale)
        normals = [[0, 0, -1], [0, 0, 1], [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]]
        tangents = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]
        bitangents = [[0, 1, 0], [0, 1, 0], [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1]]
        self._vertices = np.array(
            [
                # positions        normal        texture Coords
                [-0.5, -0.5, -0.5, *normals[0], 0.0, 0.0, *tangents[0], *bitangents[0]],
                [0.5, 0.5, -0.5, *normals[0], 1.0, 1.0, *tangents[0], *bitangents[0]],
                [0.5, -0.5, -0.5, *normals[0], 1.0, 0.0, *tangents[0], *bitangents[0]],
                [0.5, 0.5, -0.5, *normals[0], 1.0, 1.0, *tangents[0], *bitangents[0]],
                [-0.5, -0.5, -0.5, *normals[0], 0.0, 0.0, *tangents[0], *bitangents[0]],
                [-0.5, 0.5, -0.5, *normals[0], 0.0, 1.0, *tangents[0], *bitangents[0]],
                [-0.5, -0.5, 0.5, *normals[1], 0.0, 0.0, *tangents[1], *bitangents[1]],
                [0.5, -0.5, 0.5, *normals[1], 1.0, 0.0, *tangents[1], *bitangents[1]],
                [0.5, 0.5, 0.5, *normals[1], 1.0, 1.0, *tangents[1], *bitangents[1]],
                [0.5, 0.5, 0.5, *normals[1], 1.0, 1.0, *tangents[1], *bitangents[1]],
                [-0.5, 0.5, 0.5, *normals[1], 0.0, 1.0, *tangents[1], *bitangents[1]],
                [-0.5, -0.5, 0.5, *normals[1], 0.0, 0.0, *tangents[1], *bitangents[1]],
                [-0.5, 0.5, 0.5, *normals[2], 1.0, 0.0, *tangents[2], *bitangents[2]],
                [-0.5, 0.5, -0.5, *normals[2], 1.0, 1.0, *tangents[2], *bitangents[2]],
                [-0.5, -0.5, -0.5, *normals[2], 0.0, 1.0, *tangents[2], *bitangents[2]],
                [-0.5, -0.5, -0.5, *normals[2], 0.0, 1.0, *tangents[2], *bitangents[2]],
                [-0.5, -0.5, 0.5, *normals[2], 0.0, 0.0, *tangents[2], *bitangents[2]],
                [-0.5, 0.5, 0.5, *normals[2], 1.0, 0.0, *tangents[2], *bitangents[2]],
                [0.5, 0.5, 0.5, *normals[3], 1.0, 0.0, *tangents[3], *bitangents[3]],
                [0.5, -0.5, -0.5, *normals[3], 0.0, 1.0, *tangents[3], *bitangents[3]],
                [0.5, 0.5, -0.5, *normals[3], 1.0, 1.0, *tangents[3], *bitangents[3]],
                [0.5, -0.5, -0.5, *normals[3], 0.0, 1.0, *tangents[3], *bitangents[3]],
                [0.5, 0.5, 0.5, *normals[3], 1.0, 0.0, *tangents[3], *bitangents[3]],
                [0.5, -0.5, 0.5, *normals[3], 0.0, 0.0, *tangents[3], *bitangents[3]],
                [-0.5, -0.5, -0.5, *normals[4], 0.0, 1.0, *tangents[4], *bitangents[4]],
                [0.5, -0.5, -0.5, *normals[4], 1.0, 1.0, *tangents[4], *bitangents[4]],
                [0.5, -0.5, 0.5, *normals[4], 1.0, 0.0, *tangents[4], *bitangents[4]],
                [0.5, -0.5, 0.5, *normals[4], 1.0, 0.0, *tangents[4], *bitangents[4]],
                [-0.5, -0.5, 0.5, *normals[4], 0.0, 0.0, *tangents[4], *bitangents[4]],
                [-0.5, -0.5, -0.5, *normals[4], 0.0, 1.0, *tangents[4], *bitangents[4]],
                [-0.5, 0.5, -0.5, *normals[5], 0.0, 1.0, *tangents[5], *bitangents[5]],
                [0.5, 0.5, 0.5, *normals[5], 1.0, 0.0, *tangents[5], *bitangents[5]],
                [0.5, 0.5, -0.5, *normals[5], 1.0, 1.0, *tangents[5], *bitangents[5]],
                [-0.5, 0.5, -0.5, *normals[5], 0.0, 1.0, *tangents[5], *bitangents[5]],
                [-0.5, 0.5, 0.5, *normals[5], 0.0, 0.0, *tangents[5], *bitangents[5]],
                [0.5, 0.5, 0.5, *normals[5], 1.0, 0.0, *tangents[5], *bitangents[5]],
            ],
            dtype=np.float32,
        )
        _nvertices = 36
        self._indices = np.array(range(0, _nvertices), dtype=np.uint32)
        self.scale(scale).translate(position)


class Quad(Object3d):
    def __init__(self, position: NDArray[np.float32], material: Material, scale: NDArray[np.float32]):
        super().__init__(position=position, material=material, scale=scale)
        tangent = [1, 0, 0]
        bitangent = [0, 1, 0]
        normal = [0, 0, -1]
        self._vertices = np.array(
            [
                # positions      normal          uv        tangent        bitangent
                [-1.0, 1.0, 0.0, *normal, 0.0, 1.0, *tangent, *bitangent],
                [1.0, 1.0, 0.0, *normal, 1.0, 1.0, *tangent, *bitangent],
                [-1.0, -1.0, 0.0, *normal, 0.0, 0.0, *tangent, *bitangent],
                [1.0, 1.0, 0.0, *normal, 1.0, 1.0, *tangent, *bitangent],
                [1.0, -1.0, 0.0, *normal, 1.0, 0.0, *tangent, *bitangent],
                [-1.0, -1.0, 0.0, *normal, 0.0, 0.0, *tangent, *bitangent],
            ],
            dtype=np.float32,
        )
        _nvertices = 6
        self._indices = np.array([range(0, _nvertices)], dtype=np.uint32)
        self._indices = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
        self.scale(scale).translate(position)


class Ikosaeder(Object3d):
    def __init__(self, position: NDArray[np.float32], material: Material, r):
        super().__init__(position, material, scale=np.array([1, 1, 1]))
        self.r = r
        nodes, trigs = self._compute_ikosaeder_nodes_and_trigs()
        self.nodes = nodes
        self.trigs = trigs
        self._vertices = []
        self._nvertices = 0
        self._fill_vertices(nodes, trigs)

    def _compute_ikosaeder_nodes_and_trigs(self):
        angle = 2 * np.pi / 5
        # based on p0 = np.array([1,0,0]), p1 = [np.cos(angle), np.sin(angle), 0],
        # p2 = [np.cos(0.5 * angle), np.sin(0.5 * angle), h]
        p1p0 = (1 - np.cos(angle)) ** 2 + np.sin(angle) ** 2
        p2p0 = (1 - np.cos(0.5 * angle)) ** 2 + np.sin(0.5 * angle) ** 2  # + h**2
        delta = np.sqrt(p1p0 - p2p0)
        # since a**2 * p1p0 == a**2 * p2p0 + a**2 * h**2 --> h = sqrt(p1p0-p2p0) --> h = delta
        # also the point [0,0,z] must satisfy z**2 = a**2 + a**2 * (0.5 * h)**2
        # --> z**2 = a**2 * (1 + h**2) <-> a = sqrt(z/(1+h**2))
        h = delta
        a = self.r / np.sqrt(1 + 0.25 * h**2)

        nodes = np.zeros([12, 3])
        nodes[0, :] = np.array([0, 0, self.r])
        nodes[1:6, :] = np.array([a * np.array([np.cos(i * angle), np.sin(i * angle), 0.5 * h]) for i in range(5)])
        nodes[6:11, :] = np.array([a * np.array([np.cos(i * angle + 0.5 * angle), np.sin(i * angle + 0.5 * angle), -0.5 * h]) for i in range(5)])
        nodes[11, :] = np.array([0, 0, -self.r])

        trigs = [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [0, 5, 1],
            [6, 1, 2],
            [2, 6, 7],
            [7, 2, 3],
            [3, 7, 8],
            [8, 3, 4],
            [4, 8, 9],
            [9, 4, 5],
            [5, 9, 10],
            [10, 5, 1],
            [1, 10, 6],
            [11, 6, 7],
            [11, 7, 8],
            [11, 8, 9],
            [11, 9, 10],
            [11, 10, 6],
        ]

        return nodes, trigs

    def _get_tex_coord(self, node_index, trig, nodes):
        node = nodes[node_index]
        trig_x = (nodes[trig_node][0] for trig_node in trig)
        coords = 0.5 * np.array([1 + np.arctan2(node[2], node[0]) / np.pi, np.acos(-node[1] / self.r) * 2 / np.pi])
        if all(x < 0 for x in trig_x):
            if coords[0] < 0.5:
                coords[0] = 1 + coords[0]
        return coords

    def _fill_vertices(self, nodes, trigs):
        self._vertices = list(self._vertices)
        for trig in trigs:
            for node in trig:
                normalized_node = self.r * nodes[node] / np.linalg.norm(nodes[node])
                tangent = [-normalized_node[1], normalized_node[0], 0]
                if np.linalg.norm(tangent) <= 1e-12:
                    tangent = [1, 0, 0]
                position = nodes[node]
                self._vertices += (
                    list(position)
                    + list(normalized_node)
                    + list(self._get_tex_coord(node, trig, nodes))
                    + tangent
                    + [
                        -tangent[0] * normalized_node[2],
                        -tangent[1] * normalized_node[2],
                        normalized_node[0] * tangent[1] + normalized_node[1] * tangent[0],
                    ]
                )
        self._vertices = np.array(self._vertices, dtype=np.float32)
        self._nvertices += len(trigs) * 3
        self._indices = np.array([range(0, self._nvertices)], dtype=np.uint32)


class Sphere(Ikosaeder):
    def __init__(self, position=np.array([0, 0, 0]), material=Material(), r=1.0):
        super().__init__(position, material, r)
        self._split_triangles_and_fill_vertices()

    def _compute_radial_scaling_factor(self):
        node = self.nodes[0]
        diff = np.linalg.norm(self.nodes - node, axis=1)
        neighbours = np.argsort(diff)[1:6]
        # compute new center. center is on line r * (node_x, node_y, node_z),
        # and in plane node_x * x + node_y * y + node_z * z = node * (node + 1/3 * (neighbour_0 - node))
        # r * ||node||^2 = n \dot neighbour_0 -> r = 1/||node||^2 * (node + 1/3 * (neighbour_0 - node))
        # ||node|| = 1 -> r = node \dot (node + 1/3 * (neighbour_0 - node))
        n = node / np.linalg.norm(node)
        radial_scaling = np.dot(n, node + 1 / 3 * (self.nodes[neighbours[0]] - node)) / np.linalg.norm(node)
        return radial_scaling

    def _fill_local_nodes(self, trig, local_nodes, radial_scaling):
        center = 1 / 3 * (self.nodes[trig[0]] + self.nodes[trig[1]] + self.nodes[trig[2]])
        local_nodes[0] = center

        local_nodes[1] = self.nodes[trig[0]] + 1 / 3 * (self.nodes[trig[1]] - self.nodes[trig[0]])
        local_nodes[2] = self.nodes[trig[0]] + 2 / 3 * (self.nodes[trig[1]] - self.nodes[trig[0]])

        local_nodes[3] = self.nodes[trig[1]] + 1 / 3 * (self.nodes[trig[2]] - self.nodes[trig[1]])
        local_nodes[4] = self.nodes[trig[1]] + 2 / 3 * (self.nodes[trig[2]] - self.nodes[trig[1]])

        local_nodes[5] = self.nodes[trig[2]] + 1 / 3 * (self.nodes[trig[0]] - self.nodes[trig[2]])
        local_nodes[6] = self.nodes[trig[2]] + 2 / 3 * (self.nodes[trig[0]] - self.nodes[trig[2]])

        local_nodes[7] = radial_scaling * self.nodes[trig[0]]
        local_nodes[8] = radial_scaling * self.nodes[trig[1]]
        local_nodes[9] = radial_scaling * self.nodes[trig[2]]

    def _fill_local_trigs(self, local_trigs):
        n_trigs_per_trig = 9
        n_nodes_per_trig = 10
        for j in range(n_trigs_per_trig - 1):
            local_trigs[j] = np.array([0, j + 1, j + 2], dtype=np.uint8)
        local_trigs[n_trigs_per_trig - 3 - 1] = np.array([0, n_nodes_per_trig - 3 - 1, 1], dtype=np.uint8)

        local_trigs[n_trigs_per_trig - 2 - 1] = np.array([7, 1, 6])
        local_trigs[n_trigs_per_trig - 1 - 1] = np.array([8, 3, 2])
        local_trigs[n_trigs_per_trig - 0 - 1] = np.array([9, 5, 4])

    def _split_triangles_and_fill_vertices(self):
        radial_scaling = self._compute_radial_scaling_factor()
        n_trigs_per_trig = 9
        n_nodes_per_trig = 10
        nodes = np.zeros([n_nodes_per_trig * len(self.trigs), 3])
        trigs = np.zeros([n_trigs_per_trig * len(self.trigs), 3], dtype=np.uint8)
        for i, trig in enumerate(self.trigs):
            local_nodes = nodes[i * n_nodes_per_trig : (i + 1) * n_nodes_per_trig, :]
            self._fill_local_nodes(trig, local_nodes, radial_scaling)

            local_trigs = trigs[i * n_trigs_per_trig : (i + 1) * n_trigs_per_trig, :]
            self._fill_local_trigs(local_trigs)
            for trig in local_trigs:
                mat = np.array(
                    [
                        np.array(local_nodes[trig[0], :]),
                        np.array(local_nodes[trig[1], :]),
                        np.array(local_nodes[trig[2], :]),
                    ]
                )
                if np.linalg.det(mat) < 0:
                    temp = trig[1]
                    trig[1] = trig[2]
                    trig[2] = temp
            local_trigs += i * len(local_nodes)  # local to global

        self._normalize_nodes(nodes)
        self._vertices = []
        self._nvertices = 0
        self._fill_vertices(nodes, trigs)

    def _normalize_nodes(self, nodes):
        for node in nodes:
            node *= self.r / np.linalg.norm(node)


class SphericalCoordianteSphere(Object3d):
    def __init__(self, position=np.array([0, 0, 0]), material=Material(), r=1.0):
        super().__init__(position, material, scale=[1, 1, 1])
        self.r = r
        n_u = 20
        n_v = 20
        params_u = np.arange(0, 1 + 1 / n_u, 1 / n_u)
        params_v = np.arange(0, 1 + 1 / n_v, 1 / n_v)
        self._vertices: List[float] = []
        for us in zip(params_u, params_u[1:]):
            for vs in zip(params_v, params_v[1:]):
                nodes: List[List[float]] = []
                for v in vs:
                    for u in us:
                        x = np.cos(2 * np.pi * u) * np.sin(np.pi * v)
                        y = np.sin(2 * np.pi * u) * np.sin(np.pi * v)
                        z = np.cos(np.pi * v)
                        tangent = [-y, x, 0]
                        if np.linalg.norm(tangent) <= 1e-12:
                            tangent = [1, 0, 0]
                        bitangent = [-tangent[0] * z, -tangent[1] * z, x * tangent[1] + y * tangent[0]]
                        nodes.append([r * x, r * y, r * z, x, y, z, u, v, *tangent, *bitangent])
                self._vertices.extend(nodes[0])
                self._vertices.extend(nodes[3])
                self._vertices.extend(nodes[1])
                self._vertices.extend(nodes[0])
                self._vertices.extend(nodes[2])
                self._vertices.extend(nodes[3])
        _nvertices = len(self._vertices) // 14
        self._vertices = np.array(self._vertices, dtype=np.float32)
        self._indices = np.array(range(0, _nvertices), dtype=np.uint32)
