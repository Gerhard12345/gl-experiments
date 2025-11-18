from ..objects.objects3d import Object3d

from ..scenes.scene import Scene
from .shader import Shader
from OpenGL.GL import *
import numpy as np
import ctypes
from .glmaterial import GLMaterial
from numpy.typing import NDArray


class VertexBuffer:
    def __init__(self):
        self.a_position = 0
        self.a_normal = 1
        self.a_textureuv = 2
        self.a_tangent = 3
        self.a_bitangent = 4

    def upload_data_to_gpu(self, vertices: NDArray[np.float32], indices: NDArray[np.uint32]):
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        self._enable_vertex_attributes()
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def __enter__(self):
        glBindVertexArray(self.VAO)

    def __exit__(self, exc_type, exc_value, traceback):
        glBindVertexArray(0)

    def _enable_vertex_attributes(self):
        floatsize = np.dtype(np.float32).itemsize
        stride = (3 + 3 + 2 + 3 + 3) * floatsize
        glVertexAttribPointer(self.a_position, 3, GL_FLOAT, False, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(self.a_position)
        glVertexAttribPointer(self.a_normal, 3, GL_FLOAT, False, stride, ctypes.c_void_p(3 * floatsize))
        glEnableVertexAttribArray(self.a_normal)
        glVertexAttribPointer(self.a_textureuv, 2, GL_FLOAT, False, stride, ctypes.c_void_p((3 + 3) * floatsize))
        glEnableVertexAttribArray(self.a_textureuv)
        glVertexAttribPointer(self.a_tangent, 3, GL_FLOAT, False, stride, ctypes.c_void_p((3 + 3 + 2) * floatsize))
        glEnableVertexAttribArray(self.a_tangent)
        glVertexAttribPointer(self.a_bitangent, 3, GL_FLOAT, False, stride, ctypes.c_void_p((3 + 3 + 2 + 3) * floatsize))
        glEnableVertexAttribArray(self.a_bitangent)


class View:
    def __init__(self, baseobject: Object3d):
        # Initiate texture
        self.baseobject = baseobject
        self.buffer = VertexBuffer()
        self.buffer.upload_data_to_gpu(vertices=baseobject.get_vertices(), indices=baseobject.get_indices())
        self.material = GLMaterial(material=baseobject.material)
        self.element_type = GL_TRIANGLES

    def draw(self, cull_face: bool):
        if cull_face:
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
        else:
            glDisable(GL_CULL_FACE)
        with self.buffer:
            with self.material:
                glDrawElements(self.element_type, self.baseobject._nvertices, GL_UNSIGNED_INT, None)
        glDisable(GL_CULL_FACE)


class SceneView:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.viewable_objects = [View(object) for object in scene.objects]

    def draw(self, shader: Shader, cull_face=False):
        for current_object in self.viewable_objects:
            modelmat = current_object.baseobject.modelmat
            shader.setModelmat(modelmat.astype(np.float32))
            current_object.draw(cull_face=cull_face)
