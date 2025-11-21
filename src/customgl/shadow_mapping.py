import sys

import numpy as np
from OpenGL import GL

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QSurfaceFormat, QMouseEvent
from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt6.QtCore import QTimer
from PyQt6.QtCore import Qt

from .drawing.objectviews import SceneView
from .drawing.openglrenderer import (
    ShadowRenderer,
    RGBRenderer,
    PointShadowRenderer,
    Renderer,
    QuadRenderer,
    OpenGLCamera,
    CommonShaderData,
)
from .helper.windowsscaling import get_windows_scaling_factor
from .objects.camera import Camera, Camera1
from .scenes.scene import Scene, Scene1, Scene3, Scene4




# implementing a custom openGl widget
class GLWidget(QOpenGLWidget):

    def __init__(self, parent):
        QOpenGLWidget.__init__(self, parent=parent)
        self.setMinimumSize(100, 400)
        self.scene: Scene = None
        self.camera: Camera = None
        print("set up shadow renderer")
        self.shadow_renderer: Renderer = ShadowRenderer(n_lights=4)
        print("set up point shadow renderer")
        self.point_shadow_renderer: Renderer = PointShadowRenderer(n_lights=4)
        print("set up rgb renderer")
        self.rgb_renderer: Renderer = RGBRenderer(n_lights=4)
        print("set up quad renderer")
        self.quad_on_screen_renderer = QuadRenderer()
        self.opengl_camera: OpenGLCamera = None
        self.scene_view: SceneView = None
        self.last_position = None
        self.manual_camera = True
        self.do_update = False
        print("set up common shader data")
        self.common_shader_data: CommonShaderData = CommonShaderData()

    def initializeGL(self):
        print("initialize shadow renderer")
        self.shadow_renderer.initialize()
        print("initialize rgb renderer")
        self.rgb_renderer.initialize()
        print("initialize point shadow renderer")
        self.point_shadow_renderer.initialize()
        print("initialize quad renderer")
        self.quad_on_screen_renderer.initialize()
        print("create buffers")
        self.create_vertex_buffer()
        print("done")

    def paintGL(self):
        GL.glEnable(GL.GL_TEXTURE_CUBE_MAP_SEAMLESS)
        GL.glEnable(GL.GL_DEPTH_TEST)
        self.common_shader_data.prepare_omnidirectional_shader_with_transformations(
            shader=self.point_shadow_renderer.shader, omnidirectional_shadows_framebuffer=self.point_shadow_renderer.framebuffer
        )
        self.point_shadow_renderer.render(scene_view=self.scene_view)
        self.common_shader_data.prepare_directional_shader_with_transformations(
            shader=self.shadow_renderer.shader, directional_shadows_framebuffer=self.shadow_renderer.framebuffer
        )
        self.shadow_renderer.render(scene_view=self.scene_view)
        self.opengl_camera.update_camera_matrices_in_shader(
            shader=self.rgb_renderer.shader,
            viewing_width=self.rgb_renderer.width,
            viewing_height=self.rgb_renderer.height,
        )
        self.common_shader_data.prepare_rgb_shader_with_transformations_and_depth_maps(
            shader=self.rgb_renderer.shader,
            directional_shadow_framebuffer=self.shadow_renderer.framebuffer,
            omnidirectional_shadows_framebuffer=self.point_shadow_renderer.framebuffer,
        )
        self.rgb_renderer.render(scene_view=self.scene_view)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT | GL.GL_COLOR_BUFFER_BIT)
        self.quad_on_screen_renderer.render(
            shadow_texture=self.shadow_renderer.framebuffer.glrboid, rgb_texture=self.rgb_renderer.framebuffer.gltexid
        )

    def resizeGL(self, width, height):
        w = int(width * get_windows_scaling_factor())
        h = int(height * get_windows_scaling_factor())
        self.shadow_renderer.set_size(width=w, height=h)
        self.rgb_renderer.set_size(width=w, height=h)
        self.point_shadow_renderer.set_size(width=w, height=h)
        self.quad_on_screen_renderer.set_size(width=w, height=h)

    def create_vertex_buffer(self):
        print("create objects")
        self.scene = Scene4()
        # self.scene = Scene1()

        print("done")
        self.camera = Camera(eye=[0, 4, 24], at=[0, 0, 0], up=[0, 1, 0])
        # self.camera = Camera1(eye=[0, 4, 24], at=[0, 0, 0], up=[0, 1, 0])
        print("actually creating buffer")
        self.scene_view = SceneView(scene=self.scene)
        print("done")
        self.opengl_camera = OpenGLCamera(self.camera)

    def set_drawing_index(self, index: int):
        self.quad_on_screen_renderer.set_drawing_index(index)
        self.repaint()

    def unproject(self, window_x: int, window_y: int):
        self.rgb_renderer.framebuffer.bind()
        render_width, render_height = self.rgb_renderer.framebuffer.width, self.rgb_renderer.framebuffer.height
        window_x = int(window_x * get_windows_scaling_factor())
        window_y = render_height - int(window_y * get_windows_scaling_factor())
        window_z = GL.glReadPixels(window_x, window_y, 1, 1, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
        window_x = window_x / render_width * 2 - 1
        window_y = window_y / render_height * 2 - 1
        window_z = window_z[0, 0] * 2 - 1
        window_coords = np.matrix([[window_x], [window_y], [window_z], [1]])
        viewmat = self.camera.getViewmat().T
        projectionmat = self.camera.getProjectionmat(viewing_width=render_width, viewing_height=render_height).T
        outmat = projectionmat * viewmat
        outmat = outmat ** (-1)
        res = np.array(outmat * window_coords)
        print(res[:, 0] / res[3, 0])
        self.camera.set_lookat_position(res[:, 0] / res[3, 0])

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if not self.manual_camera:
            return
        self.unproject(event.pos().x(), event.pos().y())

    def wheelEvent(self, event):
        if not self.manual_camera:
            return
        scaling = 1 + (-event.angleDelta().y() // 120) * 0.25
        self.camera.zoom(scaling)

    def mouseReleaseEvent(self, _: QMouseEvent):
        if not self.manual_camera:
            return
        self.last_position = None

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.manual_camera:
            return
        if self.last_position:
            diff = [event.position().x() - self.last_position.x(), event.position().y() - self.last_position.y()]
            if event.buttons() == Qt.MouseButton.RightButton:
                self.camera.translate(diff)
            elif event.buttons() == Qt.MouseButton.LeftButton:
                self.camera.rotate_phi(diff[0])
                self.camera.rotate_theta(-diff[1])
        self.last_position = event.position()

    def update_scene(self):
        if self.do_update:
            self.scene.update()

    def update_camera(self):
        if not self.manual_camera:
            self.camera.update()

    def redraw(self):
        self.repaint()


class MyQWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)

        combobox = QComboBox()
        combobox.addItems(["Scene", "Shadow 1", "Shadow 2", "Shadow 3", "Shadow 4"])
        combobox.activated.connect(self.activated)
        layout = QVBoxLayout()
        layout.addWidget(combobox)
        button_layout = QHBoxLayout()
        button_texts = ["diffuse map", "normal map", "amb. occ. map", "specular map", "object update", "manual camera"]
        button_states = [True, True, True, True, False, True]
        button_parameters = [0, 1, 2, 3, -1, -2]
        for button_text, button_parameter, button_state in zip(button_texts, button_parameters, button_states):
            button = QPushButton(button_text)
            button.setCheckable(True)
            button.setChecked(button_state)
            button.pressed.connect(lambda val=button_parameter: self.toggle(val))
            button_layout.addWidget(button)
        layout.addLayout(button_layout)
        self.gl = GLWidget(parent=self)
        self.gl.format().setVersion(4, 2)
        self.gl.format().setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        layout.addWidget(self.gl)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.gl.update_scene)
        self.timer.timeout.connect(self.gl.update_camera)
        self.timer.timeout.connect(self.gl.redraw)
        self.timer.start(5)

    def toggle(self, value: int):
        for myobject in self.gl.scene_view.viewable_objects:
            if value == 0:
                myobject.material.texture.toggle_detailed_diffuse_maps()
            if value == 1:
                myobject.material.texture.toggle_detailed_normal_maps()
            if value == 2:
                myobject.material.texture.toggle_detailed_ambient_occlusion_maps()
            if value == 3:
                myobject.material.texture.toggle_detailed_specular_maps()
        if value == -1:
            self.gl.do_update = not self.gl.do_update
        if value == -2:
            self.gl.manual_camera = not self.gl.manual_camera

    def activated(self, index):
        self.gl.set_drawing_index(index - 1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Custom GL app")
        self.resize(600, 600)
        self.setCentralWidget(MyQWidget(self))


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
