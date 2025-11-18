import sys
import numpy as np
from PIL import Image

from OpenGL.GL import *
from OpenGL.GLU import *

from PyQt5.QtOpenGL import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QWidget, QVBoxLayout

from .drawing.shader import Shader
from .drawing.customframebuffer import CustomFrameBuffer
from .scenes.scene import Scene, Scene1, Scene2, Scene3
from .objects.camera import Camera
from .objects.objects3d import Quad
from .drawing.objectviews import VertexBuffer
from .objects.material import Material
from .drawing.projection_matrices import getCentralProjectionMatrix, getOrthogonalProjectionMatrix

SCALE = 1.0


class GLWidget(QGLWidget):

    def __init__(self, parent):
        self.parent = parent
        QGLWidget.__init__(self, parent)
        self.setMinimumSize(100, 400)
        self.scene: Scene = None
        self.framebuffer: CustomFrameBuffer = None
        self.lightspace_depth_framebuffer: CustomFrameBuffer = None
        self.drawing_index = -1

    def initialize_fullscreen_quad(self):
        shader = Shader()
        shader.compile_shader(
            "./customgl11/drawing/shaders/simple.vert",
            "./customgl11/drawing/shaders/simple.frag",
        )
        self.quad_on_screen_shader = shader
        q = Quad(position=np.array([0.0, 0.0, 0.0]), material=Material())
        self.buffer = VertexBuffer()
        self.buffer.upload_data_to_gpu(vertices=q.get_vertices(), indices=q.get_indices())

    def draw_texture_to_fullscreen_quad(self):
        self.quad_on_screen_shader.use()
        self.quad_on_screen_shader.setInt(0, "scene_texture")
        self.quad_on_screen_shader.setInt(1, "shadow_texture")
        self.quad_on_screen_shader.setInt(self.drawing_index, "shadow_component")
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.framebuffer.gltexid)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.lightspace_depth_framebuffer.glrboid)
        with self.buffer:
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

    def initialize_rgb_stuff(self):
        self.framebuffer = CustomFrameBuffer(n_lights=4)
        self.framebuffer.addColorBuffer()
        self.framebuffer.addDepthBuffer()
        shader = Shader()
        shader.add_define("N_LIGHTS", 4)
        shader.compile_shader(
            "./customgl11/drawing/shaders/main.vert",
            "./customgl11/drawing/shaders/main.frag",
        )
        self.shader = shader
        self._createVertexBuffer()

    def initialize_lightspace_depth_stuff(self):
        self.lightspace_depth_framebuffer = CustomFrameBuffer(n_lights=4)
        self.lightspace_depth_framebuffer.addMultiDepthBuffer()
        shader = Shader()
        shader.add_define("N_LIGHTS", 4)
        shader.compile_shader(
            "./customgl11/drawing/shaders/shadow.vert",
            "./customgl11/drawing/shaders/shadow.frag",
        )
        self.lightspace_depth_shader = shader

    def draw_lightspace_depth_stuff(self):
        self.lightspace_depth_shader.use()
        self.lightspace_depth_shader.setProjectionmat(getOrthogonalProjectionMatrix((self.width(), self.height())))
        for i in range(self.scene.n_lights):
            self.lightspace_depth_shader.setViewmat(self.scene.lights[i].getViewmat())
            self.lightspace_depth_framebuffer.bind(i)
            glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)
            self.scene.draw(self.lightspace_depth_shader)

    def draw_rgb_stuff(self):
        self.shader.use()
        self.scene.camera.lookAt()
        self.shader.setViewmat(self.scene.camera.getViewmat())
        self.shader.setCameraPosition(self.scene.camera.getViewingPosition())
        self.shader.setProjectionmat(
            getCentralProjectionMatrix((self.width(), self.height()), znear=0.1, zfar=100, fov=self.scene.fov)
        )
        self.shader.setLightPositions([light.getViewingPosition() for light in self.scene.lights])
        self.shader.setMatrix4fv([light.getViewmat() for light in self.scene.lights], "u_view_mat_lightspace")
        self.shader.setMatrix4f(
            getOrthogonalProjectionMatrix((self.width(), self.height())),
            "u_projection_mat_lightspace",
        )
        self.shader.setInt(0, "imagetex")
        self.shader.setInt(1, "shadowtex")
        self.lightspace_depth_framebuffer.bind_shadow_texture()
        self.framebuffer.bind(0)
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)

        self.scene.draw(self.shader)
        self.framebuffer.unbind()

    def initializeGL(self):
        self.initialize_rgb_stuff()
        self.initialize_lightspace_depth_stuff()
        self.initialize_fullscreen_quad()

    def paintGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        self.draw_lightspace_depth_stuff()
        self.draw_rgb_stuff()
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)
        self.draw_texture_to_fullscreen_quad()

    def resizeGL(self, width, height):
        print(width, height)
        w = int(width * SCALE)
        h = int(height * SCALE)
        glViewport(0, 0, w, h)
        self.framebuffer.resize((w, h))
        self.lightspace_depth_framebuffer.resize((w, h))

    def _createVertexBuffer(self):
        self.scene = Scene3()
        self.scene.setCamera(Camera(eye=[-10, 13, 20], at=[0, 6, 12], up=[0, 1, 0]))
        # self.scene = Scene1()
        # self.scene.setCamera(Camera(eye=[0, 0, 24], at=[0, 0, 0], up=[0, 1, 0]))

    def set_drawing_index(self, index: int):
        self.drawing_index = index
        self.repaint()


class MyQWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)

        combobox = QComboBox()
        combobox.addItems(["Scene", "Shadow 1", "Shadow 2", "Shadow 3", "Shadow 4"])
        combobox.activated.connect(self.activated)
        layout = QVBoxLayout()
        layout.addWidget(combobox)
        self.gl = GLWidget(parent=self)
        self.gl.format().setVersion(4, 2)
        self.gl.format().setProfile(QGLFormat.OpenGLContextProfile.CoreProfile)
        layout.addWidget(self.gl)
        self.setLayout(layout)

    def activated(self, index):
        self.gl.set_drawing_index(index - 1)


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Custom GL app")
        self.resize(600, 600)
        self.setCentralWidget(MyQWidget(self))


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    app.exec()
