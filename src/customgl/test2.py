import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt6.QtOpenGL import *
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QSurfaceFormat, QMouseEvent
from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt6.QtCore import QTimer
from PyQt6.QtCore import Qt
import sys

from .scenes.scene import Scene, Scene1, Scene3
from .objects.camera import Camera, Camera1
from .drawing.objectviews import SceneView
from .drawing.openglrenderer import (
    ShadowRenderer,
    RGBRenderer,
    PointShadowRenderer,
    Renderer,
    QuadRenderer,
    OpenGLCamera,
    prepare_rgb_shader_with_depth_map,
)

SCALE = 1.5


# implementing a custom openGl widget
class GLWidget(QOpenGLWidget):

    def __init__(self, parent):
        self.parent = parent
        QOpenGLWidget.__init__(self, parent=parent)
        self.setMinimumSize(100, 400)
        self.scene: Scene = None
        self.camera: Camera = None
        self.shadow_renderer: Renderer = ShadowRenderer(n_lights=4)
        self.point_shadow_renderer: Renderer = PointShadowRenderer(n_lights=4)
        self.rgb_renderer: Renderer = RGBRenderer(n_lights=4)
        self.quad_on_screen_renderer = QuadRenderer()
        self.opengl_camera: OpenGLCamera = None
        self.scene_view: SceneView = None
        self.last_position = None
        self.manual_camera = False
        self.do_update = True

    def initializeGL(self):
        self.shadow_renderer.initialize()
        self.rgb_renderer.initialize()
        self.point_shadow_renderer.initialize()
        self.quad_on_screen_renderer.initialize()
        self.createVertexBuffer()

    def paintGL(self):
        glEnable(GL_DEPTH_TEST)
        self.point_shadow_renderer.render(scene_view=self.scene_view)
        self.shadow_renderer.render(scene_view=self.scene_view)
        self.opengl_camera.update_camera_matrices_in_shader(
            shader=self.rgb_renderer.shader,
            viewing_width=self.rgb_renderer.width,
            viewing_height=self.rgb_renderer.height,
        )
        prepare_rgb_shader_with_depth_map(shader=self.rgb_renderer.shader, shadow_framebuffer=self.shadow_renderer.framebuffer)
        self.rgb_renderer.render(scene_view=self.scene_view)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)
        self.quad_on_screen_renderer.render(
            shadow_texture=self.shadow_renderer.framebuffer.glrboid, rgb_texture=self.rgb_renderer.framebuffer.gltexid
        )

    def resizeGL(self, width, height):
        w = int(width * SCALE)
        h = int(height * SCALE)
        self.shadow_renderer.set_size(width=w, height=h)
        self.rgb_renderer.set_size(width=w, height=h)
        self.quad_on_screen_renderer.set_size(width=w, height=h)
        self.point_shadow_renderer.set_size(width=w, height=h)

    def createVertexBuffer(self):
        # self.scene = Scene3()
        # self.camera = Camera(eye=[-10, 13, 20], at=[0, 6, 12], up=[0, 1, 0])
        self.scene = Scene1()
        self.camera = Camera1(eye=[0, 4, 24], at=[0, 0, 0], up=[0, 1, 0])
        self.scene_view = SceneView(scene=self.scene)
        self.opengl_camera = OpenGLCamera(self.camera)


class MyQWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        layout = QVBoxLayout()
        self.gl = GLWidget(parent=self)
        self.gl.format().setVersion(4, 2)
        self.gl.format().setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        layout.addWidget(self.gl)
        self.setLayout(layout)

    def toggle(self, value: int):
        pass

    def activated(self, index):
        pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Custom GL app")
        self.resize(600, 600)
        self.main_widget = MyQWidget(self)
        self.setCentralWidget(self.main_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    app.exec()
