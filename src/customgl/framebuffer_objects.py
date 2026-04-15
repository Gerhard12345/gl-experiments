from pathlib import Path
import sys

import numpy as np
from OpenGL import GL

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QWidget, QVBoxLayout


from .drawing.customframebuffer import CustomFrameBuffer
from .drawing.objectviews import VertexBuffer
from .drawing.objectviews import SceneView
from .drawing.shader import Shader
from .drawing.lights import Lights
from .helper.windowsscaling import get_windows_scaling_factor
from .objects.camera import Camera, Camera1
from .objects.material import Material
from .objects.objects3d import Quad
from .objects.transformations import getOrthogonalProjectionMatrix, getCentralProjectionMatrix

from .scenes.scene import Scene1


# implementing a custom openGl widget
class GLWidget(QOpenGLWidget):

    def __init__(self, parent, scale_factor: float):
        self.parent = parent
        QOpenGLWidget.__init__(self, parent=parent)
        self.setMinimumSize(100, 400)
        self.scene_view: SceneView = None
        self.framebuffer: CustomFrameBuffer = None
        self.lightspace_depth_framebuffer: CustomFrameBuffer = None
        self.lights: Lights = None
        self.drawing_index = -1
        self.scene = Scene1()
        self.camera: Camera = Camera1(eye=[0, 4, 24], at=[0, 0, 0], up=[0, 1, 0])
        self.shader: Shader = None
        self.buffer: VertexBuffer = None
        self.quad_on_screen_shader: Shader = None
        self.lightspace_depth_shader: Shader = None
        self.scale_factor = scale_factor

    def _create_basic_lights(self):
        # Keep this demo independent from Scene internals: create a fixed set of
        # directional lights for layered shadow rendering.
        lights = Lights()
        light_positions = [
            [1.0, 2.0, 1.0],
            [-1.0, 2.0, 1.0],
            [1.0, 2.0, -1.0],
            [-1.0, 2.0, -1.0],
        ]
        diffuse = [[1.0, 1.0, 1.0] for _ in light_positions]
        specular = [[1.0, 1.0, 1.0] for _ in light_positions]
        lights.set_directional_lights(positions=light_positions, diffuse=diffuse, specular=specular)
        self.lights = lights

    def initialize_fullscreen_quad(self):
        shader = Shader()
        shader.compile_shader(
            Path(__file__).parent.parent / "customgl/drawing/shaders/simple.vert",
            Path(__file__).parent.parent / "customgl/drawing/shaders/simple.frag",
        )
        self.quad_on_screen_shader = shader
        q = Quad(position=np.array([0.0, 0.0, 0.0]), material=Material(), scale=np.array([1, 1, 1]))
        self.buffer = VertexBuffer()
        self.buffer.upload_data_to_gpu(vertices=q.get_vertices(), indices=q.get_indices())

    def draw_texture_to_fullscreen_quad(self):
        self.quad_on_screen_shader.use()
        self.quad_on_screen_shader.setInt(0, "scene_texture")
        self.quad_on_screen_shader.setInt(1, "shadow_texture")
        self.quad_on_screen_shader.setInt(self.drawing_index, "shadow_component")
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.framebuffer.get_color_texture_id())
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(
            self.lightspace_depth_framebuffer.get_depth_texture_target(),
            self.lightspace_depth_framebuffer.get_depth_texture_id(),
        )
        with self.buffer:
            GL.glDrawElements(GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_INT, None)

    def initialize_rgb_stuff(self):
        self.framebuffer = CustomFrameBuffer.with_rgb_and_depth(n_lights=4)
        shader = Shader()
        shader.add_define("N_LIGHTS", 4)
        shader.compile_shader(
            Path(__file__).parent.parent / "customgl/drawing/shaders/simple_with_perspective.vert",
            Path(__file__).parent.parent / "customgl/drawing/shaders/simple_with_perspective.frag",
        )
        self.shader = shader
        self._createVertexBuffer()

    def initialize_lightspace_depth_stuff(self):
        self.lightspace_depth_framebuffer = CustomFrameBuffer.with_multi_depth(n_lights=4)
        shader = Shader()
        shader.add_define("N_LIGHTS", 4)
        shader.compile_shader(
            Path(__file__).parent.parent / "customgl/drawing/shaders/shadow.vert",
            Path(__file__).parent.parent / "customgl/drawing/shaders/shadow.frag",
        )
        self.lightspace_depth_shader = shader

    def draw_lightspace_depth_stuff(self):
        self.lightspace_depth_shader.use()
        self.lightspace_depth_shader.setProjectionmat(getOrthogonalProjectionMatrix((self.width(), self.height())))
        for i, light in enumerate(self.lights.lights[: self.scene.n_lights]):
            self.lightspace_depth_shader.setViewmat(light.light_space_camera.getViewmat())
            self.lightspace_depth_framebuffer.bind(i)
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT | GL.GL_COLOR_BUFFER_BIT)
            self.scene_view.draw(self.lightspace_depth_shader)

    def draw_rgb_stuff(self):
        self.shader.use()
        self.camera.lookAt()
        self.shader.setViewmat(self.camera.getViewmat())
        self.shader.setCameraPosition(self.camera.getViewingPosition())
        self.shader.setProjectionmat(getCentralProjectionMatrix((self.width(), self.height()), znear=0.1, zfar=100, fov=self.camera.fov))
        self.framebuffer.bind(0)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT | GL.GL_COLOR_BUFFER_BIT)
        self.scene_view.draw(self.shader)
        self.framebuffer.unbind()

    def initializeGL(self):
        self._create_basic_lights()
        self.initialize_rgb_stuff()
        self.initialize_lightspace_depth_stuff()
        self.initialize_fullscreen_quad()

    def paintGL(self):
        GL.glEnable(GL.GL_DEPTH_TEST)
        self.draw_rgb_stuff()
        self.draw_lightspace_depth_stuff()
        GL.glDisable(GL.GL_DEPTH_TEST)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT | GL.GL_COLOR_BUFFER_BIT)
        self.draw_texture_to_fullscreen_quad()

    def resizeGL(self, width, height):
        print(width, height)
        w = int(width * self.scale_factor)
        h = int(height * self.scale_factor)
        GL.glViewport(0, 0, w, h)
        self.framebuffer.resize((w, h))
        self.lightspace_depth_framebuffer.resize((w, h))

    def _createVertexBuffer(self):
        self.scene_view = SceneView(self.scene)

    def set_drawing_index(self, index: int):
        self.drawing_index = index
        self.repaint()


class MyQWidget(QWidget):
    def __init__(self, parent, scale_factor: float):
        super().__init__(parent=parent)

        combobox = QComboBox()
        combobox.addItems(["Scene", "Shadow 1", "Shadow 2", "Shadow 3", "Shadow 4"])
        combobox.activated.connect(self.activated)
        layout = QVBoxLayout()
        layout.addWidget(combobox)
        self.gl = GLWidget(parent=self, scale_factor=scale_factor)
        self.gl.format().setVersion(4, 2)
        self.gl.format().setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        layout.addWidget(self.gl)
        self.setLayout(layout)

    def activated(self, index):
        self.gl.set_drawing_index(index - 1)


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self, scale_factor: float):
        super().__init__()

        self.setWindowTitle("Custom GL app")
        self.resize(600, 600)
        self.setCentralWidget(MyQWidget(self, scale_factor))


def main():
    scale_factor = get_windows_scaling_factor()
    app = QApplication(sys.argv)
    window = MainWindow(scale_factor)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
