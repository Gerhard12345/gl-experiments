"""Shadow mapping demo with Qt and OpenGL."""

import logging
import sys
from pathlib import Path

import numpy as np
from OpenGL import GL

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QMouseEvent, QSurfaceFormat
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import QObject, QThread, Qt, QTimer, pyqtSignal, pyqtSlot

from .drawing.objectviews import SceneView
from .drawing.openglrenderer import (
    ShadowRenderer,
    RGBRendererWithMeshLines,
    RGBRendererWithMeshLinesIndexed,
    PointShadowRenderer,
    Renderer,
    QuadRenderer,
    OpenGLCamera,
    CommonShaderData,
)

from .helper.windowsscaling import get_windows_scaling_factor
from .objects.camera import Camera
from .scenes.scene import Scene
from .scenes.demoscenes import Scene1, Scene3
from .plugins.scenewithfemsolution import SceneWithInstancedFemSolution, SceneWithFemSolution
from .guielements.tabview import (
    CenterHighlightSplitter,
    LightingPanelConfig,
    CameraConfig,
    TopPanel,
    TabbedPanel,
    TriangleSelectionTab,
    CameraSettingsTab,
    LightingSettingsTab,
)
from .converters.lightsettingsconverter import LightSettingsConverter
from .drawing.lights import Lights
from .app_config import ShadowMappingConfig

_SCENE_CLASSES = {"Scene1": Scene1, "Scene3": Scene3, "Scene4": SceneWithInstancedFemSolution, "SceneWithFemSolution": SceneWithFemSolution}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False


class QTextEditLogHandler(logging.Handler):
    """A logging handler that appends records to a Qt text widget."""

    def __init__(self, widget: QPlainTextEdit):
        super().__init__()
        self.widget = widget
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S"))

    def emit(self, record):
        """Append a formatted log record to the text widget."""
        msg = self.format(record)
        QTimer.singleShot(0, lambda: self.widget.appendPlainText(msg))


class ScenePreparationWorker(QObject):
    """Worker object that prepares scene data off the UI thread."""

    finished = pyqtSignal(object, object, object)
    failed = pyqtSignal(str)

    def __init__(self, scene_factory, lights_factory):
        super().__init__()
        self.configure(scene_factory, lights_factory)

    def configure(self, scene_factory, lights_factory):
        """Store the factories used to prepare the scene state."""
        self.scene_factory = scene_factory
        self.lights_factory = lights_factory

    @pyqtSlot()
    def run(self):
        """Prepare the scene and emit the result to the main thread."""
        try:
            scene = self.scene_factory()
            lights = self.lights_factory()
            camera = Camera(eye=[0, 4, 24], at=[0, 0, 0], up=[0, 1, 0])
            self.finished.emit(scene, lights, camera)
        except Exception as exc:
            self.failed.emit(str(exc))


# implementing a custom openGl widget
class GLWidget(QOpenGLWidget):
    """OpenGL widget for rendering the shadow-mapping scene."""

    camera_ready = pyqtSignal(object)
    num_triangles_ready = pyqtSignal(int, int)

    def __init__(self, parent, scale_factor: float, light_config: LightingPanelConfig):
        QOpenGLWidget.__init__(self, parent=parent)
        self.setMinimumSize(500, 200)
        self.setMouseTracking(True)
        self.scene: Scene = None
        self.lights: Lights = None
        self.lights_factory: callable = None
        self.scene_factory: callable = None
        self.light_config = light_config
        self.camera: Camera = None
        logger.info("Using directional lights: %d", light_config.num_directional_lights)
        logger.info("Using point lights: %d", light_config.num_point_lights)
        self.shadow_renderer: Renderer
        self.point_shadow_renderer: Renderer
        self.rgb_renderer: Renderer
        self.quad_on_screen_renderer = QuadRenderer()
        self.opengl_camera: OpenGLCamera = None
        self.scene_view: SceneView = None
        self.last_position = None
        self.manual_camera = True
        self.do_update = False
        logger.info("set up common shader data")
        self.common_shader_data: CommonShaderData = CommonShaderData()
        self.scale_factor = scale_factor
        self.is_initalized = False
        self._scene_thread: QThread | None = None
        self._scene_worker = None
        # signal `camera_ready` is declared at class scope

    def initializeGL(self):
        """Initialize the renderers and begin scene preparation."""
        self.shadow_renderer = ShadowRenderer(n_lights=self.light_config.num_directional_lights)
        self.point_shadow_renderer = PointShadowRenderer(n_lights=self.light_config.num_point_lights)
        self.rgb_renderer = RGBRendererWithMeshLinesIndexed(
            n_lights=(self.light_config.num_directional_lights, self.light_config.num_point_lights)
        )

        self.shadow_renderer.initialize()
        self.rgb_renderer.initialize()
        self.point_shadow_renderer.initialize()
        self.quad_on_screen_renderer.initialize()
        self.prepare_scene_in_background()

    def set_lights(self, tab_defs: list):
        """Apply the light definitions to the current widget state."""
        self.lights = LightSettingsConverter(tab_defs).to_lights()

    def prepare_scene_in_background(self):
        """Prepare the scene data in a background thread."""
        scene = self.scene_factory()
        lights = self.lights_factory()
        camera = Camera(eye=[0, 4, 24], at=[0, 0, 0], up=[0, 1, 0])
        self.scene = scene
        self.lights = lights
        self.camera = camera
        logger.info("Scene prepared, creating vertex buffer")

        self._scene_thread = QThread(self)
        self._scene_worker = ScenePreparationWorker(self.scene_factory, self.lights_factory)
        self._scene_worker.moveToThread(self._scene_thread)
        self._scene_thread.started.connect(self._scene_worker.run)
        self._scene_worker.finished.connect(self.on_scene_prepared)
        self._scene_worker.failed.connect(self.on_scene_preparation_failed)
        self._scene_worker.finished.connect(self._scene_thread.quit)
        self._scene_thread.finished.connect(self._scene_worker.deleteLater)
        self._scene_thread.finished.connect(self._scene_thread.deleteLater)
        self._scene_thread.start()

    @pyqtSlot(object, object, object)
    def on_scene_prepared(self, scene, lights, camera):
        """Store the prepared scene and create the vertex buffer."""
        self.scene = scene
        self.lights = lights
        self.camera = camera
        logger.info("Scene prepared, creating vertex buffer")
        self.create_vertex_buffer()
        if scene is not None:
            try:
                num_triangles = getattr(scene, "num_triangles", None)
                selected_triangle = getattr(scene, "selected_triangle_index", None)
            except Exception:
                pass
        try:
            # emit camera ready so UI can initialize sliders
            self.num_triangles_ready.emit(num_triangles, selected_triangle)
            self.camera_ready.emit(self.camera)
        except Exception:
            # ignore if no listener is attached or signal wasn't bound normally
            pass

    @pyqtSlot(str)
    def on_scene_preparation_failed(self, message):
        """Log a scene-preparation failure."""
        logger.error("scene preparation failed: %s", message)

    def paintGL(self):
        """Render the current frame when the scene is initialized."""
        if not self.is_initalized:
            return
        GL.glEnable(GL.GL_TEXTURE_CUBE_MAP_SEAMLESS)
        GL.glEnable(GL.GL_DEPTH_TEST)
        # self.common_shader_data.prepare_omnidirectional_shader_with_transformations(
        #     shader=self.point_shadow_renderer.shader, omnidirectional_shadows_framebuffer=self.point_shadow_renderer.framebuffer
        # )
        # self.point_shadow_renderer.render(scene_view=self.scene_view, lights=self.lights)
        # self.common_shader_data.prepare_directional_shader_with_transformations(
        #     shader=self.shadow_renderer.shader, directional_shadows_framebuffer=self.shadow_renderer.framebuffer
        # )
        # self.shadow_renderer.render(scene_view=self.scene_view, lights=self.lights)
        self.opengl_camera.update_camera_matrices_in_shader(
            shader=self.rgb_renderer.shader,
            viewing_width=self.rgb_renderer.width,
            viewing_height=self.rgb_renderer.height,
        )
        # self.common_shader_data.prepare_rgb_shader_with_transformations_and_depth_maps(
        #     shader=self.rgb_renderer.shader,
        #     directional_shadow_framebuffer=self.shadow_renderer.framebuffer,
        #     omnidirectional_shadows_framebuffer=self.point_shadow_renderer.framebuffer,
        # )
        self.rgb_renderer.render(scene_view=self.scene_view, lights=self.lights)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT | GL.GL_COLOR_BUFFER_BIT)
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.quad_on_screen_renderer.render(
            shadow_texture=self.shadow_renderer.framebuffer.get_depth_texture_id(),
            rgb_texture=self.rgb_renderer.framebuffer.get_color_texture_id(),
        )

    def resizeGL(self, width, height):
        w = int(width * self.scale_factor)
        h = int(height * self.scale_factor)
        self.shadow_renderer.set_size(width=w, height=h)
        self.rgb_renderer.set_size(width=w, height=h)
        self.point_shadow_renderer.set_size(width=w, height=h)
        self.quad_on_screen_renderer.set_size(width=w, height=h)

    def create_vertex_buffer(self):
        logger.info("actually creating buffer")
        if not self.isValid():
            logger.error("create_vertex_buffer called without a valid OpenGL context")
            return
        self.makeCurrent()
        try:
            self.scene_view = SceneView(scene=self.scene)
            logger.info("scene view created")
            self.opengl_camera = OpenGLCamera(self.camera)
            self.is_initalized = True
        finally:
            self.doneCurrent()
        self.update()

    def _on_triangle_selected(self, index: int):
        if self.scene is None:
            return
        if hasattr(self.scene, "select_triangle"):
            self.scene.select_triangle(index)
            self.update()

    def set_camera_from_gui(self, camera_config: CameraConfig):
        """Apply camera settings from the GUI CameraConfig.

        The GUI exposes FOV in degrees; the internal camera expects a "fov"
        value that is passed to getCentralProjectionMatrix as tan(angle/2).
        """
        deg = camera_config.field_of_view.get("FOV")
        angle_rad = np.deg2rad(float(deg))
        fov_value = np.tan(angle_rad * 0.5)
        self.camera.fov = fov_value
        # opengl_camera uses the same Camera instance, so no extra sync needed
        self.update()

    def set_drawing_index(self, index: int):
        self.quad_on_screen_renderer.set_drawing_index(index)
        self.repaint()

    def unproject(self, window_x: int, window_y: int):
        if not self.is_initalized or not self.isValid():
            return np.array([0, 0, 0, 1])
        self.makeCurrent()
        try:
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
            physical_position = res[:, 0] / res[3, 0]
            return physical_position
        finally:
            self.doneCurrent()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if not self.manual_camera or not self.is_initalized:
            return
        physical_position = self.unproject(event.pos().x(), event.pos().y())
        self.camera.set_lookat_position(physical_position)
        logger.info("Camera look-at position set to: %s", np.asarray(physical_position).flatten().tolist())

    def wheelEvent(self, event):
        if not self.manual_camera or not self.is_initalized:
            return
        scaling = 1 + (-event.angleDelta().y() // 120) * 0.25
        self.camera.zoom(scaling)

    def mouseReleaseEvent(self, _: QMouseEvent):
        if not self.manual_camera:
            return
        self.last_position = None

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.manual_camera or not self.is_initalized:
            return
        physical_position = self.unproject(event.pos().x(), event.pos().y())
        logger.debug("Hover coordinates: %s", np.asarray(physical_position).flatten().tolist())
        if self.last_position:
            diff = [event.position().x() - self.last_position.x(), event.position().y() - self.last_position.y()]
            if event.buttons() == Qt.MouseButton.RightButton:
                self.camera.translate(diff)
            elif event.buttons() == Qt.MouseButton.LeftButton:
                self.camera.rotate_phi(diff[0])
                self.camera.rotate_theta(-diff[1])

        self.last_position = event.position()

    def update_scene(self):
        """Advance the scene when updates are enabled."""
        if self.do_update:
            self.scene.update()

    def update_camera(self):
        """Update the camera when manual navigation is disabled."""
        if not self.manual_camera:
            self.camera.update()

    def redraw(self):
        """Request a repaint."""
        self.repaint()


class MyQWidget(QWidget):
    """Main widget containing the OpenGL view and controls."""

    def __init__(self, parent, scale_factor):
        super().__init__(parent=parent)
        # Read and store config
        app_config = ShadowMappingConfig(Path(__file__).parent / "shadow_mapping.json")
        config = LightingPanelConfig(app_config.lights_data)
        # Main layout with splitter
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Create horizontal splitter for left and right columns
        splitter = CenterHighlightSplitter(Qt.Orientation.Horizontal)

        # TOP CONTROLS
        top_panel = TopPanel()
        top_panel.setFixedHeight(64)

        combobox = QComboBox()
        combobox.setFixedHeight(24)
        combobox.addItems(["Scene"] + [f"Shadow {i + 1}" for i in range(config.num_directional_lights)])
        combobox.activated.connect(self.activated)
        top_panel.add_widget(combobox, row=0)

        button_texts = ["diffuse map", "normal map", "amb. occ. map", "specular map", "object update", "manual camera"]
        button_states = [True, True, True, True, False, True]
        button_parameters = [0, 1, 2, 3, -1, -2]
        for button_text, button_parameter, button_state in zip(button_texts, button_parameters, button_states):
            button = QPushButton(button_text)
            button.setCheckable(True)
            button.setChecked(button_state)
            button.setFixedHeight(24)
            button.pressed.connect(lambda val=button_parameter: self.toggle(val))
            top_panel.add_widget(button, row=1)
        top_panel.add_stretch(row=1)

        main_layout.addWidget(top_panel)

        # LEFT COLUMN (Column 1)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(5, 5, 5, 5)

        # Row 3: GLWidget with logging widget below (horizontal splitter)
        gl_log_splitter = CenterHighlightSplitter(Qt.Orientation.Vertical)

        # Logging widget and handler are created before the GL widget so early logs are captured
        self.log_widget = QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMinimumHeight(0)
        self.log_handler = QTextEditLogHandler(self.log_widget)
        logger.addHandler(self.log_handler)

        self.gl = GLWidget(parent=self, scale_factor=scale_factor, light_config=config)
        self.gl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.gl.format().setVersion(4, 2)
        self.gl.format().setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)

        gl_log_splitter.addWidget(self.gl)
        gl_log_splitter.addWidget(self.log_widget)
        gl_log_splitter.setStretchFactor(0, 3)  # GLWidget gets more space
        gl_log_splitter.setStretchFactor(1, 1)  # Logging gets less space
        gl_log_splitter.setCollapsible(0, False)
        gl_log_splitter.setCollapsible(1, False)

        left_layout.addWidget(gl_log_splitter)

        left_panel.setLayout(left_layout)

        # RIGHT COLUMN (Column 2)
        right_panel = TabbedPanel()
        # lighting_panel = LightingControlPanel()
        self.mesh_selection_panel = TriangleSelectionTab()
        self.light_settings_tab = LightingSettingsTab()
        self.camera_settings_tab = CameraSettingsTab()
        # self.gl.set_mesh_selection_panel(self.mesh_selection_panel)
        right_panel.add_tab(self.light_settings_tab, "Lights")
        right_panel.add_tab(self.camera_settings_tab, "Camera")
        right_panel.add_tab(self.mesh_selection_panel, "Mesh")
        config.lights_loaded.connect(self.light_settings_tab.load_config)
        config.load()

        self.gl.scene_factory = lambda: _SCENE_CLASSES[app_config.scene_name]()
        self.gl.lights_factory = lambda: LightSettingsConverter(self.light_settings_tab._TAB_DEFS).to_lights()
        self.light_settings_tab.slider_changed.connect(self.gl.set_lights)
        # update camera FOV whenever the camera sliders change
        self.camera_settings_tab.slider_changed.connect(self.gl.set_camera_from_gui)

        self.mesh_selection_panel.selected_triangle_changed.connect(self._on_triangle_selected)

        self.gl.num_triangles_ready.connect(self.mesh_selection_panel.set_triangle_count_and_value)

        # when the GLWidget has prepared its scene and camera, initialize the GUI with the camera FOV
        def _on_gl_camera_ready(camera_obj):
            try:
                # interpret camera_obj.fov as either tan(angle/2) or raw radians
                fov_val = float(getattr(camera_obj, "fov", 0.0))
            except Exception:
                return
            # compute candidate angles
            try:
                angle_from_tan = 2 * np.rad2deg(np.arctan(fov_val))
            except Exception:
                angle_from_tan = None
            try:
                angle_from_rad = np.rad2deg(fov_val)
            except Exception:
                angle_from_rad = None
            # choose the plausible angle between 1 and 179 degrees
            chosen = None
            for a in (angle_from_rad, angle_from_tan):
                if a is None:
                    continue
                if 1.0 <= a <= 179.0:
                    chosen = a
                    break
            if chosen is None:
                # fallback to angle_from_tan if available
                chosen = angle_from_tan if angle_from_tan is not None else (angle_from_rad if angle_from_rad is not None else 60.0)
            val = int(round(chosen))
            self.camera_settings_tab.set_camera_config(CameraConfig(field_of_view={"FOV": val}))

        self.gl.camera_ready.connect(_on_gl_camera_ready)
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

    def toggle(self, value: int):
        """Toggle the selected rendering/material feature for the scene objects."""
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
        """Handle the combo-box selection for the shadow layer."""
        self.gl.set_drawing_index(index - 1)

    def _on_triangle_selected(self, index: int):
        if hasattr(self, "gl") and hasattr(self.gl, "_on_triangle_selected"):
            self.gl._on_triangle_selected(index)


class MainWindow(QMainWindow):
    """Main window for the shadow mapping demo."""

    def __init__(self, scale_factor):
        super().__init__()
        self.setWindowTitle("Custom GL app")
        self.resize(600, 600)
        self.setCentralWidget(MyQWidget(self, scale_factor))


def main():
    """Launch the shadow mapping demo."""
    scale_factor = get_windows_scaling_factor()
    app = QApplication(sys.argv)
    window = MainWindow(scale_factor)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
