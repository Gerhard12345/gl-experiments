import ctypes
from typing import List

import numpy as np
from numpy.typing import NDArray
from OpenGL import GL

from .glmaterial import GLMaterial
from .shader import Shader
from ..objects.objects3d import Object3d, InstancedObject3d
from ..scenes.scene import Scene


class InstancedBuffer:
    def __init__(self):
        self.ssbo: List[int] = None

    def upload_data_to_gpu(self, data: List[NDArray[np.float32]], gpu_index: List[int]):
        if self.ssbo == None:
            self.ssbo = []
        for d, index in zip(data, gpu_index):
            ssbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, ssbo)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, d.nbytes, d, GL.GL_STATIC_DRAW)
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, index, ssbo)
            self.ssbo.append(ssbo)


class VertexBuffer:
    def __init__(self):
        self.a_position = 0
        self.a_normal = 1
        self.a_textureuv = 2
        self.a_tangent = 3
        self.a_bitangent = 4
        self.vao: int = None

    def upload_data_to_gpu(self, vertices: NDArray[np.float32], indices: NDArray[np.uint32]):
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
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
        GL.glBindVertexArray(self.vao)

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


class InstancedView:
    def __init__(self, instanced_data: InstancedObject3d):
        # Initiate texture
        self.instanced_data = instanced_data
        self.baseobject = instanced_data.baseobject
        self.buffer = VertexBuffer()
        self.buffer.upload_data_to_gpu(vertices=self.baseobject.get_vertices(), indices=self.baseobject.get_indices())
        self.instanced_buffer = InstancedBuffer()
        self.instanced_buffer.upload_data_to_gpu(data=instanced_data.get_data(), gpu_index=instanced_data.get_gpu_index())
        self.material = GLMaterial(material=self.baseobject.material)
        self.element_type = GL.GL_TRIANGLES

    def draw(self, cull_face: bool):
        if cull_face:
            GL.glEnable(GL.GL_CULL_FACE)
            GL.glCullFace(GL.GL_BACK)
        else:
            GL.glDisable(GL.GL_CULL_FACE)
        with self.buffer:
            with self.material:
                GL.glDrawArraysInstanced(self.element_type, 0, self.baseobject.get_n_vertices(), self.instanced_data.get_num_instances())
        GL.glDisable(GL.GL_CULL_FACE)


class SceneView:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.viewable_objects = [View(object) for object in scene.objects]
        self.viewable_objects.extend([InstancedView(instanced_object) for instanced_object in scene.instanced_objects])

    def draw(self, shader: Shader, cull_face=False):
        for current_object in self.viewable_objects:
            modelmat = current_object.baseobject.modelmat
            shader.set_modelmat(modelmat.astype(np.float32))
            current_object.draw(cull_face=current_object.baseobject.cull_face)
