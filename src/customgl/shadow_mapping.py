import sys

import numpy as np
from OpenGL import GL

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QSurfaceFormat, QMouseEvent, QPainter, QColor
from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGridLayout, QSlider, QTabWidget, QLabel, QSplitter, QSizePolicy, QFrame, QSplitterHandle, QPlainTextEdit, QSpacerItem
from PyQt6.QtCore import QTimer, QRect, QSize
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


class CenterHighlightSplitterHandle(QSplitterHandle):
    """Custom splitter handle that highlights only in the center."""
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setMouseTracking(True)
        self.orientation = orientation
        # Set resize cursor to indicate the handle is draggable
        if orientation == Qt.Orientation.Horizontal:
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self.setCursor(Qt.CursorShape.SplitVCursor)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        
        # Get dimensions
        height = self.height()
        width = self.width()
        
        if self.orientation == Qt.Orientation.Horizontal:
            # For vertical splitter (divides left/right): highlight center third of height
            highlight_height = height // 3
            highlight_start = (height - highlight_height) // 2
            highlight_end = highlight_start + highlight_height
            highlight_rect = QRect(0, highlight_start, width, highlight_end - highlight_start)
        else:
            # For horizontal splitter (divides top/bottom): highlight center third of width
            highlight_width = width // 3
            highlight_start = (width - highlight_width) // 2
            highlight_end = highlight_start + highlight_width
            highlight_rect = QRect(highlight_start, 0, highlight_end - highlight_start, height)
        
        # Draw highlight only in the center region
        painter.fillRect(highlight_rect, QColor(0xcc, 0xcc, 0xcc))
        painter.end()
    
    def mousePressEvent(self, event):
        # Allow dragging from anywhere on the handle
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        # Ensure mouse move events work for dragging
        super().mouseMoveEvent(event)
    
    def sizeHint(self):
        # Return appropriate size for the handle
        if self.orientation == Qt.Orientation.Horizontal:
            return QSize(self.splitter().handleWidth(), -1)
        else:
            return QSize(-1, self.splitter().handleWidth())


class CenterHighlightSplitter(QSplitter):
    """Custom splitter that uses centered highlight handles."""
    def createHandle(self):
        return CenterHighlightSplitterHandle(self.orientation(), self)


# implementing a custom openGl widget
class GLWidget(QOpenGLWidget):

    def __init__(self, parent, scale_factor: float):
        QOpenGLWidget.__init__(self, parent=parent)
        self.setMinimumSize(500, 200)
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
        self.scale_factor = scale_factor

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
        w = int(width * self.scale_factor)
        h = int(height * self.scale_factor)
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
        window_x = int(window_x * self.scale_factor)
        window_y = render_height - int(window_y * self.scale_factor)
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
    def __init__(self, parent, scale_factor):
        super().__init__(parent=parent)
        
        # Main layout with splitter
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create horizontal splitter for left and right columns
        splitter = CenterHighlightSplitter(Qt.Orientation.Horizontal)
        
        # LEFT COLUMN (Column 1)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Row 1: Scene dropdown
        combobox = QComboBox()
        combobox.addItems(["Scene", "Shadow 1", "Shadow 2", "Shadow 3", "Shadow 4"])
        combobox.activated.connect(self.activated)
        left_layout.addWidget(combobox)
        
        # Row 2: Buttons
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
        left_layout.addLayout(button_layout)
        
        # Row 3: GLWidget with logging widget below (horizontal splitter)
        gl_log_splitter = CenterHighlightSplitter(Qt.Orientation.Vertical)
        
        self.gl = GLWidget(parent=self, scale_factor=scale_factor)
        self.gl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.gl.format().setVersion(4, 2)
        self.gl.format().setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        
        # Logging widget
        self.log_widget = QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMinimumHeight(0)
        
        gl_log_splitter.addWidget(self.gl)
        gl_log_splitter.addWidget(self.log_widget)
        gl_log_splitter.setStretchFactor(0, 3)  # GLWidget gets more space
        gl_log_splitter.setStretchFactor(1, 1)  # Logging gets less space
        gl_log_splitter.setCollapsible(0, False)
        gl_log_splitter.setCollapsible(1, False)
        
        left_layout.addWidget(gl_log_splitter)
        
        left_panel.setLayout(left_layout)
        
        # RIGHT COLUMN (Column 2)
        right_panel = QWidget()
        right_layout = QGridLayout()
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # Row 0: Lights dropdown
        lights_dropdown = QComboBox()
        lights_dropdown.addItems(["Light 1", "Light 2", "Light 3", "Light 4"])
        right_layout.addWidget(lights_dropdown, 0, 0)
        
        right_layout.setRowStretch(1, 0)
        
        # Row 2: Tab widget (aligned with GLWidget which starts at row 2)
        tab_widget = QTabWidget()
        
        # Tab 1: Color with 9 sliders
        color_tab = QWidget()
        color_layout = QVBoxLayout()
        color_layout.setSpacing(20)  # Add spacing between color groups
        
        # Ambient RGB (side-by-side)
        color_layout.addWidget(self._create_separator("Ambient"))
        ambient_group = self._create_rgb_group()
        color_layout.addLayout(ambient_group)
        
        # Diffuse RGB (side-by-side)
        color_layout.addWidget(self._create_separator("Diffuse"))
        diffuse_group = self._create_rgb_group()
        color_layout.addLayout(diffuse_group)
        
        # Specular RGB (side-by-side)
        color_layout.addWidget(self._create_separator("Specular"))
        specular_group = self._create_rgb_group()
        color_layout.addLayout(specular_group)
        
        color_layout.addStretch()
        color_tab.setLayout(color_layout)
        tab_widget.addTab(color_tab, "Color")
        
        # Tab 2: Geometry with position sliders
        geometry_tab = QWidget()
        geometry_layout = QVBoxLayout()
        geometry_layout.setSpacing(20)
        
        # Ambient direction sliders (X, Y, Z)
        geometry_layout.addWidget(self._create_separator("Ambient direction"))
        ambient_direction_config = {"X": (0, 100), "Y": (0, 100), "Z": (0, 100)}
        ambient_direction_group = self._create_rgb_group(slider_config=ambient_direction_config)
        geometry_layout.addLayout(ambient_direction_group)
        
        # Point light position sliders (X, Y, Z)
        geometry_layout.addWidget(self._create_separator("Point light position"))
        point_light_config = {"X": (0, 100), "Y": (0, 100), "Z": (0, 100)}
        point_light_group = self._create_rgb_group(slider_config=point_light_config)
        geometry_layout.addLayout(point_light_group)
        
        geometry_layout.addStretch()
        geometry_tab.setLayout(geometry_layout)
        tab_widget.addTab(geometry_tab, "Geometry")
        
        right_layout.addWidget(tab_widget, 2, 0)
        right_layout.setRowStretch(2, 1)  # Tab widget row expands to fill height
        right_panel.setLayout(right_layout)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)  # Left column gets more space
        splitter.setStretchFactor(1, 1)  # Right column gets less space
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.gl.update_scene)
        self.timer.timeout.connect(self.gl.update_camera)
        self.timer.timeout.connect(self.gl.redraw)
        self.timer.start(5)
    
    def _create_separator(self, title=None):
        """Create a horizontal separator line with optional title."""
        if title is None:
            separator = QFrame()
            separator.setFrameShape(QFrame.Shape.HLine)
            separator.setFrameShadow(QFrame.Shadow.Sunken)
            return separator
        else:
            separator_layout = QHBoxLayout()
            separator_layout.setContentsMargins(0, 0, 0, 0)
            separator_layout.setSpacing(10)
            
            # Left line segment
            separator_left = QFrame()
            separator_left.setFrameShape(QFrame.Shape.HLine)
            separator_left.setFrameShadow(QFrame.Shadow.Sunken)
            separator_left.setMinimumWidth(20)
            separator_layout.addWidget(separator_left, 0)
            
            # Title label
            label = QLabel(title)
            label.setStyleSheet("font-weight: bold;")
            separator_layout.addWidget(label)
            
            # Right line segment
            separator_right = QFrame()
            separator_right.setFrameShape(QFrame.Shape.HLine)
            separator_right.setFrameShadow(QFrame.Shadow.Sunken)
            separator_layout.addWidget(separator_right, 1)
            
            separator_widget = QWidget()
            separator_widget.setLayout(separator_layout)
            return separator_widget
    
    def _create_rgb_group(self, slider_config=None, min_val=0, max_val=255):
        """Create a vertical layout with sliders for RGB/position values.
        
        Args:
            slider_config: Optional dict mapping slider labels to (min, max) tuples. 
                          If None, defaults to RGB (R, G, B) with range 0-255
            min_val: Default minimum value (used if slider_config is None)
            max_val: Default maximum value (used if slider_config is None)
        """
        group_layout = QVBoxLayout()
        
        # Use provided config or default to RGB
        if slider_config is None:
            labels = ["R", "G", "B"]
        else:
            labels = list(slider_config.keys())
        
        sliders_layout = QHBoxLayout()
        sliders_layout.setSpacing(20)
        
        for lbl in labels:
            slider_layout = QVBoxLayout()
            slider = QSlider(Qt.Orientation.Vertical)
            
            if slider_config is None:
                slider.setMinimum(min_val)
                slider.setMaximum(max_val)
            else:
                min_range, max_range = slider_config[lbl]
                slider.setMinimum(min_range)
                slider.setMaximum(max_range)
            
            slider.setValue(50)
            slider_layout.addWidget(slider)
            label_widget = QLabel(lbl)
            label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            slider_layout.addWidget(label_widget, alignment=Qt.AlignmentFlag.AlignCenter)
            sliders_layout.addLayout(slider_layout)
        
        # Center the sliders horizontally
        centered_layout = QHBoxLayout()
        centered_layout.addStretch()
        centered_layout.addLayout(sliders_layout)
        centered_layout.addStretch()
        group_layout.addLayout(centered_layout)
        
        return group_layout
    
    def _create_three_vertical_sliders(self, labels, min_val=0, max_val=100):
        """Create 3 vertical sliders side-by-side with labels. (Deprecated - use _create_rgb_group instead)"""
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)  # Add horizontal spacing between sliders
        
        for label_text in labels:
            slider_layout = QVBoxLayout()
            slider = QSlider(Qt.Orientation.Vertical)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(50)
            slider_layout.addWidget(slider)
            label = QLabel(label_text)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            slider_layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignCenter)
            main_layout.addLayout(slider_layout)
        
        return main_layout

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
    def __init__(self, scale_factor):
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
