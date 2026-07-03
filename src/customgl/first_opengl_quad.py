"""A minimal OpenGL example that draws a quad and a cube."""

import sys
from pathlib import Path

import numpy as np
from OpenGL import GL

from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from .drawing.objectviews import View
from .drawing.shader import Shader
from .objects.material import Material
from .objects.objects3d import Cube, Quad


class GLWidget(QOpenGLWidget):
    """Minimal OpenGL widget for rendering basic geometry."""

    def __init__(self, parent):
        self.parent = parent
        QOpenGLWidget.__init__(self, parent)
        self.setMinimumSize(100, 400)
        self.vq: View = None
        self.vc: View = None
        self.shader: Shader = None

    def initializeGL(self):
        """Initialize the demo geometry and shader."""
        texture_path = Path(__file__).parent.parent / "customgl" / "textures" / "testing.png"
        quad = Quad(
            position=np.array([0, 0, 0]),
            material=Material(texturefilename=texture_path),
            scale=np.array([1, 1, 1]),
        )
        cube = Cube(
            position=np.array([0, 0, 0]),
            material=Material(texturefilename=texture_path),
            scale=np.array([1, 1, 1]),
        )
        self.vq = View(quad)
        self.vc = View(cube)
        self.shader = Shader()
        self.shader.compile_shader(
            vertex_code_file=Path(__file__).parent.parent / "customgl" / "drawing/shaders/simple_with_perspective.vert",
            fragment_code_file=Path(__file__).parent.parent / "customgl" / "drawing/shaders/simple_with_perspective.frag",
        )

    def resizeGL(self, w: int, h: int) -> None:
        """Resize the viewport."""
        GL.glViewport(0, 0, w, h)

    def paintGL(self):
        """Render the current frame."""
        GL.glClearColor(0.0, 1.0, 1.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self.shader.use()
        # self.vc.draw(cull_face=False)
        self.vq.draw(cull_face=False)
        GL.glUseProgram(0)


class MyQWidget(QWidget):
    """Simple container widget for the OpenGL demo."""

    def __init__(self, parent):
        super().__init__(parent=parent)

        layout = QVBoxLayout()
        self.gl = GLWidget(parent=self)
        self.gl.format().setVersion(4, 2)
        self.gl.format().setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        layout.addWidget(self.gl)
        self.setLayout(layout)

    def get_gl_widget(self) -> GLWidget:
        """Return the embedded OpenGL widget."""
        return self.gl


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    """Main application window for the OpenGL demo."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Custom GL app")
        self.resize(600, 600)
        self.setCentralWidget(MyQWidget(self))

    def get_container_widget(self) -> MyQWidget:
        """Return the main container widget."""
        return self.centralWidget()


def main():
    """Launch the demo window."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
