import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Union
from numbers import Number

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QSlider,
    QLabel,
    QFrame,
    QSplitter,
    QSplitterHandle,
    QComboBox,
)
from PyQt6.QtCore import Qt, QSize, pyqtSlot
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPalette


@dataclass
class CameraConfig:
    field_of_view: Dict[str, int]


@dataclass
class SliderGroupDef:
    name: str
    slider_config: Dict[str, Tuple[int, int]]
    default_values: List[int]
    orientation: Qt.Orientation = Qt.Orientation.Vertical
    step_size: float = 1.0


@dataclass
class LightDef:
    name: str
    type: str  # "Point", "Directional", "Ambient"
    light_properties: Dict[str, SliderGroupDef]


_RGB_SLIDER_CONFIG = {"R": (0, 255), "G": (0, 255), "B": (0, 255)}
_VECTOR_COMPONENTS = ("X", "Y", "Z")


def _build_slider_config(prop_name: str, prop_data: dict) -> Dict[str, Tuple[int, int]]:
    if "slider_config" in prop_data:
        return {k: tuple(v) for k, v in prop_data["slider_config"].items()}

    if prop_name in {"Color", "Diffuse", "Specular"}:
        return dict(_RGB_SLIDER_CONFIG)

    if "ranges" in prop_data:
        return {component: tuple(prop_data["ranges"][component.lower()]) for component in _VECTOR_COMPONENTS}

    raise KeyError(f"Missing slider configuration for light property '{prop_name}'")


class CenterHighlightSplitterHandle(QSplitterHandle):
    """Custom splitter handle that highlights only in the center third."""

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setMouseTracking(True)
        if orientation == Qt.Orientation.Horizontal:
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self.setCursor(Qt.CursorShape.SplitVCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, self.palette().color(QPalette.ColorRole.Window))

        if self.orientation() == Qt.Orientation.Horizontal:
            # Highlight center third vertically
            center_y = rect.height() / 2
            third_height = rect.height() / 6  # One third on each side of center
            highlight_rect = rect.adjusted(0, int(center_y - third_height), 0, int(-(rect.height() - center_y - third_height)))
            painter.fillRect(highlight_rect, QColor("#cccccc"))
        else:
            # Highlight center third horizontally
            center_x = rect.width() / 2
            third_width = rect.width() / 6  # One third on each side of center
            highlight_rect = rect.adjusted(int(center_x - third_width), 0, int(-(rect.width() - center_x - third_width)), 0)
            painter.fillRect(highlight_rect, QColor("#cccccc"))

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)

    def sizeHint(self):
        if self.orientation() == Qt.Orientation.Horizontal:
            return QSize(5, 100)
        else:
            return QSize(100, 5)


class CenterHighlightSplitter(QSplitter):
    """Custom splitter with highlighted center-only handles."""

    def createHandle(self):
        return CenterHighlightSplitterHandle(self.orientation(), self)


class LightingControlPanel(QWidget):
    """
    Panel containing color and geometry controls in tabbed interface.
    """

    slider_changed = pyqtSignal(list)

    _CAMERA_GROUP = SliderGroupDef("Field of View", {"FOV": (1, 179)}, [60])

    def __init__(self, parent=None):
        super().__init__(parent)
        self._TAB_DEFS: List[LightDef] = []
        self._current_light: str = ""
        self._loading: bool = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.tab_widget = QTabWidget()
        # slider lookup: light_name -> prop_name -> label -> QSlider
        self._sliders: Dict[str, Dict[str, Dict[str, QSlider]]] = {}
        self._camera_sliders: Dict[str, Dict[str, QSlider]] = {}

        # --- Lights tab ---
        lights_tab = QWidget()
        lights_outer = QVBoxLayout()
        lights_outer.setContentsMargins(5, 5, 5, 5)
        lights_outer.setSpacing(5)
        lights_tab.setLayout(lights_outer)

        self._dropdown = QComboBox()
        self._dropdown.currentTextChanged.connect(self.switch_light)
        lights_outer.addWidget(self._dropdown)

        self._rgb_tab_layout = QVBoxLayout()
        self._geometry_tab_layout = QVBoxLayout()
        lights_outer.addLayout(self._rgb_tab_layout)
        lights_outer.addLayout(self._geometry_tab_layout)
        lights_outer.addStretch()

        self.tab_widget.addTab(lights_tab, "Lights")

        # --- Camera tab ---
        self._build_camera_tab()

        layout.addWidget(self.tab_widget)
        self.setLayout(layout)

    def _build_camera_tab(self) -> None:
        tab = QWidget()
        tab_layout = self._add_layout_to_tab(tab)
        self._camera_sliders[self._CAMERA_GROUP.name] = self._create_slider_group(
            tab_layout,
            self._CAMERA_GROUP.name,
            slider_config=self._CAMERA_GROUP.slider_config,
            default_values=self._CAMERA_GROUP.default_values,
        )
        tab_layout.addStretch()
        self.tab_widget.addTab(tab, "Camera")

    _RGB_KEYS = frozenset({"R", "G", "B"})

    def _tab_layout_for(self, group_def: SliderGroupDef) -> QVBoxLayout:
        """Route a SliderGroupDef to the RGB or Geometry tab layout."""
        if set(group_def.slider_config.keys()) <= self._RGB_KEYS:
            return self._rgb_tab_layout
        return self._geometry_tab_layout

    def _clear_light_tabs(self) -> None:
        """Remove all widgets from both light tab layouts."""
        for layout in (self._rgb_tab_layout, self._geometry_tab_layout):
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                elif item.layout():
                    self._delete_layout(item.layout())
        self._sliders.clear()

    def _delete_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._delete_layout(item.layout())

    def _populate_light_tabs(self, light_def: LightDef) -> None:
        """Fill RGB and Geometry tabs with sliders for the given LightDef."""
        if light_def.name not in self._sliders:
            self._sliders[light_def.name] = {}
        for prop_name, group_def in light_def.light_properties.items():
            if prop_name not in self._sliders[light_def.name]:
                self._sliders[light_def.name][prop_name] = self._create_slider_group(
                    self._tab_layout_for(group_def),
                    prop_name,
                    slider_config=group_def.slider_config,
                    default_values=group_def.default_values,
                    step_size=group_def.step_size,
                )

    @pyqtSlot(list)
    def load_config(self, data: List[dict]) -> None:
        self._TAB_DEFS = []
        self._sliders = {}
        self._dropdown.blockSignals(True)
        self._dropdown.clear()

        for light_dict in data:
            light_def = LightDef(
                name=light_dict["name"],
                type=light_dict["type"],
                light_properties={
                    prop_name: SliderGroupDef(
                        name=prop_name,
                        slider_config=_build_slider_config(prop_name, prop_data),
                        default_values=list(prop_data["default_values"]),
                        step_size=float(prop_data.get("step_size", 1.0)),
                    )
                    for prop_name, prop_data in light_dict["light_properties"].items()
                },
            )
            self._TAB_DEFS.append(light_def)
            self._dropdown.addItem(light_def.name)

        self._dropdown.blockSignals(False)
        if self._TAB_DEFS:
            self.switch_light(self._TAB_DEFS[0].name)

    def switch_light(self, selected_light: str) -> None:
        if not selected_light:
            return
        self._save_current_light_state()
        self._clear_light_tabs()
        light_def = next((l for l in self._TAB_DEFS if l.name == selected_light), None)
        if light_def is None:
            return
        self._loading = True
        try:
            self._populate_light_tabs(light_def)
            self._load_light_state(light_def)
        finally:
            self._loading = False
        self._current_light = selected_light

    def _save_current_light_state(self) -> None:
        if not self._current_light or self._current_light not in self._sliders:
            return
        light_def = next((l for l in self._TAB_DEFS if l.name == self._current_light), None)
        if light_def is None:
            return
        for prop_name, group_def in light_def.light_properties.items():
            group_def.default_values = [
                self._sliders[self._current_light][prop_name][lbl].value() * group_def.step_size for lbl in group_def.slider_config
            ]

    def _load_light_state(self, light_def: LightDef) -> None:
        for prop_name, group_def in light_def.light_properties.items():
            for lbl, val in zip(group_def.slider_config.keys(), group_def.default_values):
                self._sliders[light_def.name][prop_name][lbl].setValue(round(val / group_def.step_size))

    def _on_slider_changed(self, _: int) -> None:
        if self._loading:
            return
        self._save_current_light_state()
        self.slider_changed.emit(self._TAB_DEFS)

    @property
    def camera_config(self) -> CameraConfig:
        return CameraConfig(
            field_of_view={lbl: self._camera_sliders[self._CAMERA_GROUP.name][lbl].value() for lbl in self._CAMERA_GROUP.slider_config}
        )

    def set_camera_config(self, config: CameraConfig) -> None:
        for lbl, val in config.field_of_view.items():
            self._camera_sliders[self._CAMERA_GROUP.name][lbl].setValue(val)

    def _add_layout_to_tab(self, tab: QWidget) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        tab.setLayout(layout)
        return layout

    def _create_slider_group(
        self,
        layout,
        title: str,
        slider_config: Dict[str, Tuple[Number, Number]],
        default_values: List[Number],
        orientation=Qt.Orientation.Horizontal,
        step_size: float = 1.0,
    ) -> Dict[str, QSlider]:
        """
        Add a separator and a group of sliders to the given layout.
        Args:
            layout: The layout to which the separator and sliders will be added.
            title: The title for the separator.
            slider_config: dict mapping slider labels to (min, max) tuples.
            default_values: Default values for sliders.
            orientation: Qt.Orientation.Horizontal (default) or Qt.Orientation.Vertical
        """
        # Separator
        separator_layout = QHBoxLayout()
        separator_layout.setContentsMargins(0, 0, 0, 0)
        separator_layout.setSpacing(10)
        separator_left = QFrame()
        separator_left.setFrameShape(QFrame.Shape.HLine)
        separator_left.setFrameShadow(QFrame.Shadow.Sunken)
        separator_left.setMinimumWidth(20)
        separator_layout.addWidget(separator_left, 0)
        label = QLabel(title)
        label.setStyleSheet("font-weight: bold;")
        separator_layout.addWidget(label)
        separator_right = QFrame()
        separator_right.setFrameShape(QFrame.Shape.HLine)
        separator_right.setFrameShadow(QFrame.Shadow.Sunken)
        separator_layout.addWidget(separator_right, 1)
        separator_widget = QWidget()
        separator_widget.setLayout(separator_layout)
        layout.addWidget(separator_widget)

        # Sliders
        sliders: Dict[str, QSlider] = {}
        if orientation == Qt.Orientation.Horizontal:
            sliders_layout = QVBoxLayout()
            sliders_layout.setSpacing(4)
            for lbl, default_value in zip(slider_config.keys(), default_values):
                row = QHBoxLayout()
                label_widget = QLabel(lbl)
                label_widget.setFixedWidth(24)
                label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
                slider = QSlider(Qt.Orientation.Horizontal)
                min_range, max_range = slider_config[lbl]
                slider.setMinimum(round(min_range / step_size))
                slider.setMaximum(round(max_range / step_size))
                slider.setValue(round(default_value / step_size))
                value_label = QLabel(str(round(default_value)))
                value_label.setFixedWidth(40)
                value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                slider.valueChanged.connect(self._on_slider_changed)
                slider.valueChanged.connect(lambda value, label=value_label, scale=step_size: label.setText(str(round(value * scale))))
                sliders[lbl] = slider
                row.addWidget(label_widget)
                row.addWidget(slider)
                row.addWidget(value_label)
                sliders_layout.addLayout(row)
            layout.addLayout(sliders_layout)
        else:
            sliders_layout = QHBoxLayout()
            sliders_layout.setSpacing(20)
            for lbl, default_value in zip(slider_config.keys(), default_values):
                slider_layout = QVBoxLayout()
                slider = QSlider(Qt.Orientation.Vertical)
                min_range, max_range = slider_config[lbl]
                slider.setMinimum(round(min_range / step_size))
                slider.setMaximum(round(max_range / step_size))
                slider.setValue(round(default_value / step_size))
                value_label = QLabel(str(round(default_value)))
                value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                slider.valueChanged.connect(self._on_slider_changed)
                slider.valueChanged.connect(lambda value, label=value_label, scale=step_size: label.setText(str(round(value * scale))))
                sliders[lbl] = slider
                slider_layout.addWidget(slider)
                slider_layout.addWidget(value_label, alignment=Qt.AlignmentFlag.AlignCenter)
                label_widget = QLabel(lbl)
                label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
                slider_layout.addWidget(label_widget, alignment=Qt.AlignmentFlag.AlignCenter)
                sliders_layout.addLayout(slider_layout)
            centered_layout = QHBoxLayout()
            centered_layout.addStretch()
            centered_layout.addLayout(sliders_layout)
            centered_layout.addStretch()
            layout.addLayout(centered_layout)
        return sliders


class LightingPanelConfig(QObject):
    """
    Loads a light config JSON and emits lights_loaded(list) with one signal.
    Connect to the load_config slot of LightingControlPanel.
    """

    lights_loaded = pyqtSignal(list)

    def __init__(self, source: Union[Path, list], parent=None):
        super().__init__(parent)
        if isinstance(source, list):
            self._data: list = source
        else:
            with open(source, "r") as f:
                self._data: list = json.load(f)

        self.num_directional_lights: int = sum(1 for light in self._data if light.get("type") == "Directional")
        self.num_point_lights: int = sum(1 for light in self._data if light.get("type") == "Point")

    def load(self) -> None:
        self.lights_loaded.emit(self._data)
