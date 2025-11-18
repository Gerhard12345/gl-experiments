from pathlib import Path
import numpy as np

from OpenGL.GL import *
from pathlib import Path
from .shader import Shader
from .customframebuffer import CustomFrameBuffer
from .objectviews import SceneView
from .objectviews import VertexBuffer
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
        shader.setViewmat(self.camera.getViewmat())
        shader.setCameraPosition(self.camera.getViewingPosition())
        shader.setProjectionmat(self.camera.getProjectionmat(viewing_width=viewing_width, viewing_height=viewing_height))


class CommonShaderData:
    def __init__(self):
        self.omnidirectional_shadows_fov = np.pi * 0.5
        self.omnidirectional_shadows_near = 1
        self.omnidirectional_shadows_far = 50
        self.omnidirectional_shadows_texture_unit = 5

        self.directional_shadows_bottom = -28
        self.directional_shadows_top = 28
        self.directional_shadows_near = -5
        self.directional_shadows_far = 45
        self.directional_shadows_texture_unit = 0

    def prepare_rgb_shader_with_transformations_and_depth_maps(
        self, shader: Shader, directional_shadow_framebuffer: CustomFrameBuffer, omnidirectional_shadows_framebuffer: CustomFrameBuffer
    ):
        shader.use()
        shader.setMatrix4f(
            getOrthogonalProjectionMatrix(
                asize=(directional_shadow_framebuffer.width, directional_shadow_framebuffer.height),
                bottom=self.directional_shadows_bottom,
                top=self.directional_shadows_top,
                znear=self.directional_shadows_near,
                zfar=self.directional_shadows_far,
            ),
            "u_projection_mat_lightspace",
        )
        shader.setInt(self.directional_shadows_texture_unit, "shadowtex")
        shader.setInt(self.omnidirectional_shadows_texture_unit, "depthMap")
        shader.setFloat(self.omnidirectional_shadows_far, "far_plane")
        directional_shadow_framebuffer.bind_shadow_texture()
        omnidirectional_shadows_framebuffer.bind_shadow_texture()

    def prepare_omnidirectional_shader_with_transformations(self, shader: Shader, omnidirectional_shadows_framebuffer: CustomFrameBuffer):
        shader.use()
        shader.setProjectionmat(
            getCentralProjectionMatrix(
                (omnidirectional_shadows_framebuffer.width, omnidirectional_shadows_framebuffer.height),
                znear=self.omnidirectional_shadows_near,
                zfar=self.omnidirectional_shadows_far,
                fov=np.tan(self.omnidirectional_shadows_fov * 0.5),
            )
        )
        shader.setFloat(self.omnidirectional_shadows_far, "far_plane")

    def prepare_directional_shader_with_transformations(self, shader: Shader, directional_shadows_framebuffer: CustomFrameBuffer):
        shader.use()
        shader.setProjectionmat(
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
        print(Path(__file__).parent)
        print("!!!!!!!!!!!")
        self.base_directory: Path = Path(__file__).parent.parent
        self.shader_directory: Path = self.base_directory / "drawing" / "shaders"

    def initialize(self):
        pass

    def set_size(self, width: int, height: int):
        self.width = width
        self.height = height
        self.framebuffer.resize((width, height))

    def render(self, scene_view: SceneView):
        pass


class ShadowRenderer(Renderer):
    def __init__(self, n_lights: int):
        super().__init__(n_lights)
        self.framebuffer: CustomFrameBuffer = None
        self.shader: Shader = None

    def initialize(self):
        self.framebuffer = CustomFrameBuffer(n_lights=self.n_lights)
        self.framebuffer.addMultiDepthBuffer()
        shader = Shader()
        shader.add_define("N_DIRECTIONAL_LIGHTS", self.n_lights)
        shader.compile_shader(self.shader_directory / "shadow.vert", self.shader_directory / "shadow.frag")
        self.shader = shader

    def render(self, scene_view: SceneView):
        scene = scene_view.scene
        self.shader.use()
        glViewport(0, 0, self.width, self.height)
        self.shader.setProjectionmat(getOrthogonalProjectionMatrix((self.width, self.height)))
        for i in range(scene.n_lights):
            self.shader.setViewmat(scene.lights[i].light_space_camera.getViewmat())
            self.framebuffer.bind(i)
            glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
            scene_view.draw(self.shader, cull_face=True)
            glDisable(GL_CULL_FACE)


class PointShadowRenderer(Renderer):
    def __init__(self, n_lights: int):
        super().__init__(n_lights)
        self.framebuffer: CustomFrameBuffer = None
        self.shader: Shader = None

    def initialize(self):
        self.framebuffer = CustomFrameBuffer(n_lights=self.n_lights)
        self.framebuffer.addCubeMapDepthBuffer()
        shader = Shader()
        shader.add_define("N_POINT_LIGHTS", 1)
        shader.compile_shader(
            vertex_code_file=self.shader_directory / "point_shadow.vert",
            fragment_code_file=self.shader_directory / "point_shadow.frag",
            geometry_code_file=self.shader_directory / "point_shadow.geom",
        )
        self.shader = shader

    def render(self, scene_view: SceneView):
        scene = scene_view.scene
        self.shader.use()
        self.framebuffer.bind()
        glViewport(0, 0, self.width, self.height)
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)
        for i in range(self.n_lights):
            self.shader.setMatrix4fv(
                [light_space_camera.getViewmat() for light_space_camera in scene.point_lights[i].light_space_camera], uniform_name="u_view_mat"
            )
            self.shader.setVec3fv([scene.point_lights[i].position], uniform_name="lightPos")
            self.shader.setInt(i, "light_index")
            self.framebuffer.bind()
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
            scene_view.draw(self.shader, cull_face=True)
            glDisable(GL_CULL_FACE)

    def set_size(self, width: int, height: int):
        larger_dim = np.max([width, height])
        self.width = larger_dim
        self.height = larger_dim
        self.framebuffer.resize((larger_dim, larger_dim))


class RGBRenderer(Renderer):
    def __init__(self, n_lights: int):
        super().__init__(n_lights)
        self.framebuffer: CustomFrameBuffer = None
        self.shader: Shader = None

    def initialize(self):
        self.framebuffer = CustomFrameBuffer(n_lights=self.n_lights)
        self.framebuffer.addColorBuffer()
        self.framebuffer.addDepthBuffer()
        shader = Shader()
        shader.add_define("N_DIRECTIONAL_LIGHTS", 1)
        shader.add_define("N_POINT_LIGHTS", self.n_lights)
        shader.compile_shader(self.shader_directory / "main.vert", self.shader_directory / "main.frag")
        self.shader = shader

    def render(self, scene_view: SceneView):
        scene = scene_view.scene
        self.shader.use()
        glViewport(0, 0, self.width, self.height)
        self.shader.setLightPositions([light.light_space_camera.getViewingPosition() for light in scene.lights])
        self.shader.setMatrix4fv([light.light_space_camera.getViewmat() for light in scene.lights], "u_view_mat_lightspace")

        for i, light in enumerate(scene.lights):
            self.shader.setVec3fv([light.direction], f"u_directional_lights[{i}].direction")
            self.shader.setVec3fv([light.ambient], f"u_directional_lights[{i}].ambient")
            self.shader.setVec3fv([light.diffuse], f"u_directional_lights[{i}].diffuse")
            self.shader.setVec3fv([light.specular], f"u_directional_lights[{i}].specular")
        for i, light in enumerate(scene.point_lights):
            self.shader.setVec3fv([light.position], f"u_point_lights[{i}].position")
            self.shader.setVec3fv([light.ambient], f"u_point_lights[{i}].ambient")
            self.shader.setVec3fv([light.diffuse], f"u_point_lights[{i}].diffuse")
            self.shader.setVec3fv([light.specular], f"u_point_lights[{i}].specular")
            self.shader.setFloat(light.constant, f"u_point_lights[{i}].constant")
            self.shader.setFloat(light.linear, f"u_point_lights[{i}].linear")
            self.shader.setFloat(light.quadratic, f"u_point_lights[{i}].qudratic")

        self.framebuffer.bind(0)
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)

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
        glViewport(0, 0, self.width, self.height)
        self.shader.use()
        self.shader.setInt(0, "shadow_texture")
        self.shader.setInt(1, "scene_texture")
        self.shader.setInt(self.drawing_index, "shadow_component")
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, shadow_texture)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, rgb_texture)
        with self.buffer:
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

    def set_drawing_index(self, index: int):
        self.drawing_index = index

    def set_size(self, width: int, height: int):
        self.width = width
        self.height = height
