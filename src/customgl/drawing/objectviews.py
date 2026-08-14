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
        self._buffer_data: List[NDArray[np.generic]] = []

    def upload_data_to_gpu(self, data: List[NDArray[np.float32]], gpu_index: List[int]):
        if self.ssbo == None:
            self.ssbo = []
        for d, index in zip(data, gpu_index):
            ssbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, ssbo)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, d.nbytes, d, GL.GL_STATIC_DRAW)
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, index, ssbo)
            self.ssbo.append(ssbo)
            self._buffer_data.append(np.asarray(d).copy())

    def replace_buffer_element(self, buffer_indices: List[int], element_indices: List[int], values_list: List[NDArray[np.generic]]):
        """Replace a single logical SSBO element (e.g. a vec4 row) at element_index."""
        for buffer_index, element_index, values in zip(buffer_indices, element_indices, values_list):
            values = np.asarray(values)
            if buffer_index < 0 or buffer_index >= len(self.ssbo):
                raise IndexError(f"buffer_index {buffer_index} out of range for {len(self.ssbo)} buffers")
            if values.size == 0:
                return

            source_data = self._buffer_data[buffer_index]
            if source_data.ndim == 1:
                if element_index < 0 or element_index >= source_data.size:
                    raise IndexError(f"element_index {element_index} out of range for 1D buffer of size {source_data.size}")
                payload = np.asarray(values, dtype=source_data.dtype).reshape(-1)
                offset = element_index * source_data.dtype.itemsize
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.ssbo[buffer_index])
                GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, offset, payload.nbytes, payload)
                self._buffer_data[buffer_index][element_index] = payload[0]
                return

            row_size = source_data.shape[1]
            if element_index < 0 or element_index >= source_data.shape[0]:
                raise IndexError(f"element_index {element_index} out of range for buffer of size {source_data.shape[0]}")

            payload = np.asarray(values, dtype=source_data.dtype)
            if payload.ndim == 0:
                payload = payload.reshape(1)
            if payload.size != row_size:
                raise ValueError(f"Expected {row_size} values to replace a row, got {payload.size}")

            row = payload.reshape(1, row_size)
            offset = element_index * source_data.dtype.itemsize * row_size
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.ssbo[buffer_index])
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, offset, row.nbytes, row)
            self._buffer_data[buffer_index][element_index] = row[0]

    def replace_buffer_range(self, buffer_index: int, start_index: int, values: NDArray[np.generic]):
        """Replace a contiguous range of logical SSBO elements in one call."""
        values = np.asarray(values)
        if buffer_index < 0 or buffer_index >= len(self.ssbo):
            raise IndexError(f"buffer_index {buffer_index} out of range for {len(self.ssbo)} buffers")
        source_data = self._buffer_data[buffer_index]

        if values.size == 0:
            return

        if source_data.ndim == 1:
            if start_index < 0:
                raise IndexError("start_index must be >= 0")
            payload = np.asarray(values, dtype=source_data.dtype).reshape(-1)
            if start_index + payload.size > source_data.size:
                raise IndexError(f"Range [{start_index}, {start_index + payload.size}) exceeds buffer size {source_data.size}")
            offset = start_index * source_data.dtype.itemsize
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.ssbo[buffer_index])
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, offset, payload.nbytes, payload)
            self._buffer_data[buffer_index][start_index : start_index + payload.size] = payload
            return

        if start_index < 0 or start_index >= source_data.shape[0]:
            raise IndexError(f"start_index {start_index} out of range for buffer of size {source_data.shape[0]}")

        rows = np.asarray(values, dtype=source_data.dtype)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.shape[1] != source_data.shape[1]:
            raise ValueError(f"Expected row width {source_data.shape[1]}, got {rows.shape[1]}")

        count = rows.shape[0]
        if start_index + count > source_data.shape[0]:
            raise IndexError(f"Range [{start_index}, {start_index + count}) exceeds buffer size {source_data.shape[0]}")

        offset = start_index * source_data.dtype.itemsize * source_data.shape[1]
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.ssbo[buffer_index])
        GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, offset, rows.nbytes, rows)
        self._buffer_data[buffer_index][start_index : start_index + count] = rows


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
        if self.instanced_data.shall_update:
            self.instanced_buffer.replace_buffer_element(
                self.instanced_data.buffer_index, self.instanced_data.element_index, self.instanced_data.values
            )
            self.instanced_buffer.shall_update = False
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
