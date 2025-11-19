import ctypes

import numpy as np
from numpy.typing import NDArray
import OpenGL.GL as GL

from .glmaterial import GLMaterial
from .shader import Shader
from ..objects.objects3d import Object3d
from ..scenes.scene import Scene


class VertexBuffer:
    def __init__(self):
        self.a_position = 0
        self.a_normal = 1
        self.a_textureuv = 2
        self.a_tangent = 3
        self.a_bitangent = 4
        self.VAO: int = None

    def upload_data_to_gpu(self, vertices: NDArray[np.float32], indices: NDArray[np.uint32]):
        self.VAO = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.VAO)
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
        ibo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ibo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)
        self._enable_vertex_attributes()
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def __enter__(self):
        GL.glBindVertexArray(self.VAO)

    def __exit__(self, exc_type, exc_value, traceback):
        GL.glBindVertexArray(0)

    def _enable_vertex_attributes(self):
        floatsize = np.dtype(np.float32).itemsize
        stride = (3 + 3 + 2 + 3 + 3) * floatsize
        GL.glVertexAttribPointer(self.a_position, 3, GL.GL_FLOAT, False, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(self.a_position)
        GL.glVertexAttribPointer(self.a_normal, 3, GL.GL_FLOAT, False, stride, ctypes.c_void_p(3 * floatsize))
        GL.glEnableVertexAttribArray(self.a_normal)
        GL.glVertexAttribPointer(self.a_textureuv, 2, GL.GL_FLOAT, False, stride, ctypes.c_void_p((3 + 3) * floatsize))
        GL.glEnableVertexAttribArray(self.a_textureuv)
        GL.glVertexAttribPointer(self.a_tangent, 3, GL.GL_FLOAT, False, stride, ctypes.c_void_p((3 + 3 + 2) * floatsize))
        GL.glEnableVertexAttribArray(self.a_tangent)
        GL.glVertexAttribPointer(self.a_bitangent, 3, GL.GL_FLOAT, False, stride, ctypes.c_void_p((3 + 3 + 2 + 3) * floatsize))
        GL.glEnableVertexAttribArray(self.a_bitangent)


class View:
    def __init__(self, baseobject: Object3d):
        # Initiate texture
        self.baseobject = baseobject
        self.buffer = VertexBuffer()
        self.buffer.upload_data_to_gpu(vertices=baseobject.get_vertices(), indices=baseobject.get_indices())
        self.material = GLMaterial(material=baseobject.material)
        self.element_type = GL.GL_TRIANGLES

    def draw(self, cull_face: bool):
        if cull_face:
            GL.glEnable(GL.GL_CULL_FACE)
            GL.glCullFace(GL.GL_BACK)
        else:
            GL.glDisable(GL.GL_CULL_FACE)
        with self.buffer:
            with self.material:
                GL.glDrawElements(self.element_type, self.baseobject.get_n_vertices(), GL.GL_UNSIGNED_INT, None)
        GL.glDisable(GL.GL_CULL_FACE)


class SceneView:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.viewable_objects = [View(object) for object in scene.objects]

    def draw(self, shader: Shader, cull_face=False):
        for current_object in self.viewable_objects:
            modelmat = current_object.baseobject.modelmat
            shader.setModelmat(modelmat.astype(np.float32))
            current_object.draw(cull_face=cull_face)
