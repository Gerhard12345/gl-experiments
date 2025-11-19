import sys
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import *
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt6.QtGui import QSurfaceFormat

from .objects.objects3d import Quad, Cube
from .objects.material import Material
from .drawing.objectviews import View
from .drawing.shader import Shader


# implementing a custom openGl widget
class GLWidget(QOpenGLWidget):

    def __init__(self, parent):
        self.parent = parent
        QOpenGLWidget.__init__(self, parent)
        self.setMinimumSize(100, 400)
        self.vq: View = None
        self.vc: View = None
        self.shader: Shader = None

    def initializeGL(self):
        quad = Quad(
            position=np.array([0, 0, 0]), material=Material(texturefilename="./customgl/textures/testing.png"), scale=np.array([1, 1, 1])
        )
        cube = Cube(
            position=np.array([0, 0, 0]), material=Material(texturefilename="./customgl/textures/testing.png"), scale=np.array([1, 1, 1])
        )
        self.vq = View(quad)
        self.vc = View(cube)
        self.shader = Shader()
        self.shader.compile_shader(
            vertex_code_file="./customgl/drawing/shaders/simple_with_perspective.vert",
            fragment_code_file="./customgl/drawing/shaders/simple_with_perspective.frag",
        )

    def resizeGL(self, w: int, h: int) -> None:
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClearColor(0.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.shader.use()
        #self.vc.draw(cull_face=False)
        self.vq.draw(cull_face=False)
        glUseProgram(0)


class MyQWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)

        layout = QVBoxLayout()
        self.gl = GLWidget(parent=self)
        self.gl.format().setVersion(4, 2)
        self.gl.format().setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        layout.addWidget(self.gl)
        self.setLayout(layout)


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
