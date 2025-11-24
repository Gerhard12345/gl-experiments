import numpy as np
import pytest

from src.customgl.objects import objects3d

# Use small local dummy implementations to mock dependencies from the material and
# transformations modules inside objects3d.  This keeps the tests focused on
# objects3d behaviour and not on the real dependencies.


class DummyMaterial:
    def __init__(self, **kwargs):
        self.specularpower = kwargs.get("specularpower", 32)
        self.texture_scales = kwargs.get("texture_scales", [1, 1])


class DummyTransformations:
    @staticmethod
    def scalemat(scale_xyz: np.ndarray):
        # return a predictable matrix dependent on input
        return np.matrix(
            [[scale_xyz[0], 0, 0, 0], [0, scale_xyz[1], 0, 0], [0, 0, scale_xyz[2], 0], [0, 0, 0, 1]]
        ).transpose()

    @staticmethod
    def translationmat(position: np.ndarray):
        return np.matrix([[1, 0, 0, position[0]], [0, 1, 0, position[1]], [0, 0, 1, position[2]], [0, 0, 0, 1]]).transpose()

    @staticmethod
    def rotationmat_axis(angle: float, axis: np.ndarray):
        # return a simple deterministic matrix that changes when angle/axis change
        return np.matrix([[1 + angle, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    @staticmethod
    def localrotationmat_axis(position, angle, axis):
        return DummyTransformations.translationmat(-position) * DummyTransformations.rotationmat_axis(angle, axis) * DummyTransformations.translationmat(position)


Object3d = objects3d.Object3d
Cube = objects3d.Cube
Quad = objects3d.Quad
Ikosaeder = objects3d.Ikosaeder
Sphere = objects3d.Sphere
SphericalCoordianteSphere = objects3d.SphericalCoordianteSphere


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # patch the module-level references inside objects3d so the classes created there
    # will use our dummy implementations instead of the real material/transformations
    monkeypatch.setattr(objects3d, "Material", DummyMaterial)
    monkeypatch.setattr(objects3d, "Transformations", DummyTransformations)


def test_object3d_basic_transforms():
    pos = np.array([1.0, 2.0, 3.0])
    mat = objects3d.Material()
    scale = np.array([1.0, 1.0, 1.0])

    o = Object3d(position=pos, material=mat, scale=scale)

    # modelmat should be the identity matrix initially
    assert np.allclose(np.asarray(o.modelmat), np.identity(4))

    # scale should multiply model matrix
    o.scale(np.array([2.0, 3.0, 4.0]))
    expected = objects3d.Transformations.scalemat(np.array([2.0, 3.0, 4.0]))
    assert np.allclose(np.asarray(o.modelmat), np.asarray(expected))

    # rotate and translate should modify the model matrix and return the object
    old = np.array(o.modelmat.copy())
    o.rotate_x(0.2)
    assert isinstance(o, Object3d)
    assert not np.allclose(np.asarray(o.modelmat), np.asarray(old))

    old = np.array(o.modelmat.copy())
    o.translate(np.array([0.5, -0.5, 1.0]))
    assert not np.allclose(np.asarray(o.modelmat), np.asarray(old))


def test_cube_and_quad_vertices_indices():
    mat = objects3d.Material()
    c = Cube(position=np.array([0.1, 0.2, 0.3]), material=mat, scale=np.array([0.5, 0.5, 0.5]))
    v = c.get_vertices()
    idx = c.get_indices()

    # Cube uses 36 vertices and each vertex has 14 floats (pos,norm,uv,tan,bitan)
    assert v.shape[0] == 36
    assert v.shape[1] == 14
    assert idx.size == 36
    assert c.get_n_trigs() == 12  # 36 vertices -> 12 triangles
    assert c.get_n_vertices() == 36

    q = Quad(position=np.array([0, 0, 0]), material=mat, scale=np.array([1, 1, 1]))
    vq = q.get_vertices()
    idxq = q.get_indices()
    assert vq.shape[0] == 6
    assert vq.shape[1] == 14
    # indices for quad are two triangles -> 6 indices total
    assert idxq.size == 6
    assert q.get_n_trigs() == 2


def test_ikosaeder_nodes_trigs_and_get_texcoord():
    ik = Ikosaeder(position=np.array([0.0, 0.0, 0.0]), material=objects3d.Material(), r=1.0)

    nodes, trigs = ik._compute_ikosaeder_nodes_and_trigs()
    # expect 12 nodes and 20 triangle faces
    assert nodes.shape == (12, 3)
    assert len(trigs) == 20

    # Test internal texture coordinate wrapping behaviour:
    # build a small nodes array where x < 0 for all nodes in trig and choose a node
    # that produces coords[0] < 0.5 so it should be wrapped to coords[0] + 1
    nodes_small = np.array([[-1.0, 0.0, -1e-6], [-0.7, 0.0, 0.0], [-0.6, 0.0, 0.0]])
    trig = [0, 1, 2]
    coords = ik._get_tex_coord(0, trig, nodes_small)
    # when x negative and coords[0] was below 0.5 it'll be wrapped into [1,2)
    assert coords[0] >= 1.0

    # After initialization, Ikosaeder should have vertex and index arrays consistent with number of triangles
    assert ik._nvertices == len(trigs) * 3
    assert ik._indices.size == ik._nvertices
    assert isinstance(ik._vertices, np.ndarray)


def test_sphere_and_spherical_coordinate_sphere_build():
    # basic sphere should refine the icosaeder and produce more vertices
    base = Ikosaeder(position=np.array([0, 0, 0]), material=objects3d.Material(), r=1.0)
    s = Sphere(position=np.array([0, 0, 0]), material=objects3d.Material(), r=1.0)

    assert s._nvertices > base._nvertices
    assert s._vertices.dtype == np.float32
    assert s._indices.size == s._nvertices

    scs = SphericalCoordianteSphere(position=np.array([0, 0, 0]), material=objects3d.Material(), r=2.0)
    # vertices are stored as 14 floats each and indices length should equal number of vertices
    assert scs._vertices.dtype == np.float32
    assert scs._vertices.size % 14 == 0
    assert scs._indices.size == scs._vertices.size // 14
