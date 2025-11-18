import enum
from pathlib import Path
from OpenGL.GL import *
from numpy.typing import NDArray
from typing import List

# from kivy.graphics.opengl import *

import numpy as np
from typing import Dict


class Shader:
    program = None

    def __init__(self):
        self.program: int = None
        self.defines: Dict[str, int | float] = {}

    def add_define(self, name_to_define: str, value):
        self.defines[name_to_define] = value

    def _create_and_compile_shader(self, shader_source_file: Path, shader_type: int):
        with open(shader_source_file) as f:
            code = f.read()
        for name_to_define, value in self.defines.items():
            temp = code.index("#version")
            index = code.index("\n", temp)
            code = code[: index + 1] + f"#define {name_to_define} {value}\n" + code[index + 1 :]
        shader = glCreateShader(shader_type)
        glShaderSource(shader, bytes(code, "utf-8"))
        glCompileShader(shader)
        return shader

    def compile_shader(self, vertex_code_file: Path, fragment_code_file: Path, geometry_code_file: Path = None) -> None:
        ## shader construction

        program = glCreateProgram()

        if vertex_code_file is not None:
            vertex_shader = self._create_and_compile_shader(vertex_code_file, GL_VERTEX_SHADER)
            glAttachShader(program, vertex_shader)
        if fragment_code_file is not None:
            fragment_shader = self._create_and_compile_shader(fragment_code_file, GL_FRAGMENT_SHADER)
            glAttachShader(program, fragment_shader)
        if geometry_code_file is not None:
            geometry_shader = self._create_and_compile_shader(geometry_code_file, GL_GEOMETRY_SHADER)
            glAttachShader(program, geometry_shader)

        glLinkProgram(program)

        if vertex_code_file is not None:
            glDetachShader(program, vertex_shader)
        if fragment_code_file is not None:
            glDetachShader(program, fragment_shader)
        if geometry_code_file is not None:
            glDetachShader(program, geometry_shader)

        self.program = program

    def use(self):
        glUseProgram(self.program)

    def setViewmat(self, viewMatrix):
        self.setMatrix4fv([viewMatrix], "u_view_mat")

    def setCameraPosition(self, cameraPosition):
        self.setVec3fv([cameraPosition], "u_viewing_position")

    def setLightPositions(self, positions):
        self.setVec3fv(positions, "u_light_position")

    def setVec3fv(self, vecs: List[NDArray[np.float32]], uniform_name):
        u_viewing_pos = glGetUniformLocation(self.program, bytes(uniform_name, "utf-8"))
        glUniform3fv(u_viewing_pos, len(vecs), np.array(vecs).astype(np.float32).flatten().tobytes())

    def setModelmat(self, modelmat: NDArray[np.float32]):
        self.setMatrix4fv([modelmat], "u_model_mat")

    def setProjectionmat(self, projectionmat: NDArray[np.float32]):
        self.setMatrix4fv([projectionmat], "u_projection_mat")

    def setMatrix4f(self, matrix4f: NDArray[np.float32], uniform_name: str):
        u_mat = glGetUniformLocation(self.program, bytes(uniform_name, "utf-8"))
        glUniformMatrix4fv(u_mat, 1, False, np.array([matrix4f]).flatten().tobytes())

    def setMatrix4fv(self, matrix4f: List[NDArray[np.float32]], uniform_name: str):
        size = len(matrix4f)
        u_mat = glGetUniformLocation(self.program, bytes(uniform_name, "utf-8"))
        glUniformMatrix4fv(u_mat, size, False, np.concatenate(matrix4f).flatten().tobytes())

    def setInt(self, value: int, uniform_name: str):
        u_int = glGetUniformLocation(self.program, bytes(uniform_name, "utf-8"))
        glUniform1i(u_int, value)

    def setFloat(self, value: float, uniform_name: str):
        u_float = glGetUniformLocation(self.program, bytes(uniform_name, "utf-8"))
        glUniform1f(u_float, value)
