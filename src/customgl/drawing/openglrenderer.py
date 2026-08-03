from pathlib import Path
from typing import Tuple

import numpy as np
from OpenGL import GL

from .customframebuffer import CustomFrameBuffer
from .objectviews import SceneView
from .objectviews import VertexBuffer
from .shader import Shader
from .lights import Lights
from ..objects.transformations import getOrthogonalProjectionMatrix, getCentralProjectionMatrix
from ..objects.material import Material
from ..objects.objects3d import Quad
from ..objects.camera import Camera


class OpenGLCamera:
    def __init__(self, camera: Camera):
        self.camera = camera

    def update_camera_matrices_in_shader(self, shader: Shader, viewing_width: int, viewing_height: int):
        shader.use()
        self.camera.lookAt()
        shader.set_viewmat(self.camera.getViewmat())
        shader.set_camera_position(self.camera.getViewingPosition())
        shader.set_projection_mat(self.camera.getProjectionmat(viewing_width=viewing_width, viewing_height=viewing_height))


class CommonShaderData:
    def __init__(self):
        self.omnidirectional_shadows_fov = np.pi * 0.5
        self.omnidirectional_shadows_near = 1
        self.omnidirectional_shadows_far = 50
        self.omnidirectional_shadows_texture_unit = 1

        self.directional_shadows_bottom = -28
        self.directional_shadows_top = 28
        self.directional_shadows_near = -5
        self.directional_shadows_far = 45
        self.directional_shadows_texture_unit = 0

    def prepare_rgb_shader_with_transformations_and_depth_maps(
        self, shader: Shader, directional_shadow_framebuffer: CustomFrameBuffer, omnidirectional_shadows_framebuffer: CustomFrameBuffer
    ):
        shader.use()
        shader.set_matrix_4f(
            getOrthogonalProjectionMatrix(
                asize=(directional_shadow_framebuffer.width, directional_shadow_framebuffer.height),
                bottom=self.directional_shadows_bottom,
                top=self.directional_shadows_top,
                znear=self.directional_shadows_near,
                zfar=self.directional_shadows_far,
            ),
            "u_projection_mat_lightspace",
        )
        shader.set_int(self.directional_shadows_texture_unit, "directional_shadow_map")
        shader.set_int(self.omnidirectional_shadows_texture_unit, "depthMap")
        shader.set_float(self.omnidirectional_shadows_far, "far_plane")
        GL.glActiveTexture(GL.GL_TEXTURE0 + self.omnidirectional_shadows_texture_unit)
        GL.glBindTexture(
            omnidirectional_shadows_framebuffer.get_depth_texture_target(),
            omnidirectional_shadows_framebuffer.get_depth_texture_id(),
        )
        GL.glActiveTexture(GL.GL_TEXTURE0 + self.directional_shadows_texture_unit)
        GL.glBindTexture(
            directional_shadow_framebuffer.get_depth_texture_target(),
            directional_shadow_framebuffer.get_depth_texture_id(),
        )

    def prepare_omnidirectional_shader_with_transformations(self, shader: Shader, omnidirectional_shadows_framebuffer: CustomFrameBuffer):
        shader.use()
        shader.set_projection_mat(
            getCentralProjectionMatrix(
                (omnidirectional_shadows_framebuffer.width, omnidirectional_shadows_framebuffer.height),
                znear=self.omnidirectional_shadows_near,
                zfar=self.omnidirectional_shadows_far,
                fov=np.tan(self.omnidirectional_shadows_fov * 0.5),
            )
        )
        shader.set_float(self.omnidirectional_shadows_far, "far_plane")

    def prepare_directional_shader_with_transformations(self, shader: Shader, directional_shadows_framebuffer: CustomFrameBuffer):
        shader.use()
        shader.set_projection_mat(
            getOrthogonalProjectionMatrix(
                asize=(directional_shadows_framebuffer.width, directional_shadows_framebuffer.height),
                bottom=self.directional_shadows_bottom,
                top=self.directional_shadows_top,
                znear=self.directional_shadows_near,
                zfar=self.directional_shadows_far,
            )
        )


class Renderer:
    def __init__(self, n_lights: int):
        self.n_lights = n_lights
        self.width: int = 0
        self.height: int = 0
        self.framebuffer: CustomFrameBuffer = None
        self.shader: Shader = None
        self.base_directory: Path = Path(__file__).parent.parent
        self.shader_directory: Path = self.base_directory / "drawing" / "shaders"

    def initialize(self):
        pass

    def set_size(self, width: int, height: int):
        self.width = width
        self.height = height
        self.framebuffer.resize((width, height))

    def render(self, scene_view: SceneView, lights: Lights = None):
        pass


class ShadowRenderer(Renderer):
    def __init__(self, n_lights: int):
        super().__init__(n_lights)
        self.framebuffer: CustomFrameBuffer = None
        self.shader: Shader = None

    def initialize(self):
        self.framebuffer = CustomFrameBuffer.with_multi_depth(n_layers=self.n_lights)
        shader = Shader()
        shader.add_define("N_DIRECTIONAL_LIGHTS", self.n_lights)
        shader.compile_shader(self.shader_directory / "shadow.vert", self.shader_directory / "shadow.frag")
        self.shader = shader

    def render(self, scene_view: SceneView, lights: Lights = None):
        self.shader.use()
        GL.glViewport(0, 0, self.width, self.height)
        self.shader.set_projection_mat(getOrthogonalProjectionMatrix((self.width, self.height)))
        directional_lights = lights.lights[: self.n_lights]
        for i, directional_light in enumerate(directional_lights):
            self.shader.set_viewmat(directional_light.light_space_camera.getViewmat())
            self.framebuffer.bind(i)
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT | GL.GL_COLOR_BUFFER_BIT)
            GL.glEnable(GL.GL_CULL_FACE)
            GL.glCullFace(GL.GL_BACK)
            scene_view.draw(self.shader, cull_face=True)
            GL.glDisable(GL.GL_CULL_FACE)


class PointShadowRenderer(Renderer):
    def __init__(self, n_lights: int):
        super().__init__(n_lights)
        self.framebuffer: CustomFrameBuffer = None
        self.shader: Shader = None

    def initialize(self):
        self.framebuffer = CustomFrameBuffer.with_cubemap_depth(n_lights=self.n_lights)
        shader = Shader()
        shader.add_define("N_POINT_LIGHTS", self.n_lights)
        shader.compile_shader(
            vertex_code_file=self.shader_directory / "point_shadow.vert",
            fragment_code_file=self.shader_directory / "point_shadow.frag",
            geometry_code_file=self.shader_directory / "point_shadow.geom",
        )
        self.shader = shader

    def render(self, scene_view: SceneView, lights: Lights = None):
        self.shader.use()
        self.framebuffer.bind()
        GL.glViewport(0, 0, self.width, self.height)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT | GL.GL_COLOR_BUFFER_BIT)
        point_lights = lights.point_lights[: self.n_lights]
        for i, point_light in enumerate(point_lights):
            self.shader.set_matrix_4fv(
                [light_space_camera.getViewmat() for light_space_camera in point_light.light_space_camera], uniform_name="u_view_mat"
            )
            self.shader.set_vec_3fv([point_light.position], uniform_name="lightPos")
            self.shader.set_int(i, "light_index")
            self.framebuffer.bind()
            GL.glEnable(GL.GL_CULL_FACE)
            GL.glCullFace(GL.GL_BACK)
            scene_view.draw(self.shader, cull_face=True)
            GL.glDisable(GL.GL_CULL_FACE)

    def set_size(self, width: int, height: int):
        larger_dim = np.max([width, height])
        self.width = larger_dim
        self.height = larger_dim
        self.framebuffer.resize((larger_dim, larger_dim))


class RGBRenderer(Renderer):
    def __init__(self, n_lights: Tuple[int, int]):
        super().__init__(n_lights[0] + n_lights[1])
        self.framebuffer: CustomFrameBuffer = None
        self.shader: Shader = None
        self.n_directional_lights = n_lights[0]
        self.n_point_lights = n_lights[1]

    def initialize(self):
        self.framebuffer = CustomFrameBuffer.with_rgb_and_depth()
        shader = Shader()
        shader.add_define("N_DIRECTIONAL_LIGHTS", self.n_directional_lights)
        shader.add_define("N_POINT_LIGHTS", self.n_point_lights)
        shader.compile_shader(self.shader_directory / "main.vert", self.shader_directory / "main.frag")
        self.shader = shader

    def render(self, scene_view: SceneView, lights: Lights = None):
        self.shader.use()
        GL.glViewport(0, 0, self.width, self.height)
        directional_lights = lights.lights[: self.n_directional_lights]
        point_lights = lights.point_lights[: self.n_point_lights]

        if directional_lights:
            self.shader.set_light_positions([light.light_space_camera.getViewingPosition() for light in directional_lights])
            self.shader.set_matrix_4fv([light.light_space_camera.getViewmat() for light in directional_lights], "u_view_mat_lightspace")

        self.shader.set_vec_3fv([lights.ambient_light.color], "u_ambient_light.color")
        for i, light in enumerate(directional_lights):
            self.shader.set_vec_3fv([light.direction], f"u_directional_lights[{i}].direction")
            self.shader.set_vec_3fv([light.diffuse], f"u_directional_lights[{i}].diffuse")
            self.shader.set_vec_3fv([light.specular], f"u_directional_lights[{i}].specular")
        for i, light in enumerate(point_lights):
            self.shader.set_vec_3fv([light.position], f"u_point_lights[{i}].position")
            self.shader.set_vec_3fv([light.diffuse], f"u_point_lights[{i}].diffuse")
            self.shader.set_vec_3fv([light.specular], f"u_point_lights[{i}].specular")
            self.shader.set_float(light.constant, f"u_point_lights[{i}].constant")
            self.shader.set_float(light.linear, f"u_point_lights[{i}].linear")
            self.shader.set_float(light.quadratic, f"u_point_lights[{i}].qudratic")

        self.framebuffer.bind(0)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT | GL.GL_COLOR_BUFFER_BIT)

        scene_view.draw(self.shader, cull_face=True)
        self.framebuffer.unbind()


class QuadRenderer:
    def __init__(self):
        self.base_directory: Path = Path(__file__).parent.parent
        self.shader_directory: Path = self.base_directory / "drawing" / "shaders"
        self.width: int = None
        self.height: int = None
        self.buffer: VertexBuffer = None
        self.shader: Shader = None
        self.drawing_index: int = -1

    def initialize(self):
        self.shader = Shader()
        self.shader.compile_shader(self.shader_directory / "simple.vert", self.shader_directory / "simple.frag")
        q = Quad(position=np.array([0.0, 0.0, 0.0]), scale=np.array([1, 1, 1]), material=Material())
        self.buffer = VertexBuffer()
        self.buffer.upload_data_to_gpu(vertices=q.get_vertices(), indices=q.get_indices())

    def render(self, shadow_texture: int, rgb_texture: int):
        GL.glViewport(0, 0, self.width, self.height)
        self.shader.use()
        self.shader.set_int(0, "shadow_texture")
        self.shader.set_int(1, "scene_texture")
        self.shader.set_int(self.drawing_index, "shadow_component")
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, shadow_texture)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, rgb_texture)
        with self.buffer:
            GL.glDrawElements(GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_INT, None)

    def set_drawing_index(self, index: int):
        self.drawing_index = index

    def set_size(self, width: int, height: int):
        self.width = width
        self.height = height


class RGBRendererWithMeshLines(RGBRenderer):
    def initialize(self):
        self.framebuffer = CustomFrameBuffer.with_rgb_and_depth()
        shader = Shader()
        shader.add_define("N_DIRECTIONAL_LIGHTS", self.n_directional_lights)
        shader.add_define("N_POINT_LIGHTS", self.n_point_lights)
        shader.compile_shader(
            self.shader_directory / "main_highlighting_triangle_bounds.vert", self.shader_directory / "main_highlighting_triangle_bounds.frag"
        )
        self.shader = shader
