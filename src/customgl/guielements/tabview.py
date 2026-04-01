from dataclasses import dataclass
from typing import Dict, Tuple, List
from numbers import Number

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget, QSlider,
    QLabel, QPushButton, QFrame, QPlainTextEdit, QSplitter, QSpacerItem,
    QSizePolicy, QSplitterHandle
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPainter, QColor, QPalette


@dataclass
class ColorConfig:
    ambient: Dict[str, int]
    diffuse: Dict[str, int]
    specular: Dict[str, int]


@dataclass
class GeometryConfig:
    ambient_direction: Dict[str, int]
    point_light_position: Dict[str, int]


@dataclass
class CameraConfig:
    field_of_view: Dict[str, int]


@dataclass
class SliderGroupDef:
    name: str
    slider_config: Dict[str, Tuple[int, int]]
    default_values: List[int]
    orientation: Qt.Orientation = Qt.Orientation.Vertical


@dataclass
class TabDef:
    name: str
    slider_groups: List[SliderGroupDef]


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

    _TAB_DEFS: List[TabDef] = [
        TabDef("Color", [
            SliderGroupDef("Ambient",         {"R": (0, 255), "G": (0, 255), "B": (0, 255)}, [128, 128, 128]),
            SliderGroupDef("Diffuse",         {"R": (0, 255), "G": (0, 255), "B": (0, 255)}, [128, 128, 128]),
            SliderGroupDef("Specular",        {"R": (0, 255), "G": (0, 255), "B": (0, 255)}, [128, 128, 128]),
        ]),
        TabDef("Geometry", [
            SliderGroupDef("Ambient direction",    {"X": (0, 100), "Y": (0, 100), "Z": (0, 100)}, [50, 50, 50]),
            SliderGroupDef("Point light position", {"X": (0, 100), "Y": (0, 100), "Z": (0, 100)}, [50, 50, 50]),
        ]),
        TabDef("Camera Settings", [
            SliderGroupDef("Field of View", {"FOV": (1, 179)}, [60]),
        ]),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create the tab widget
        self.tab_widget = QTabWidget()
        self._sliders: Dict[str, Dict[str, Dict[str, QSlider]]] = {}

        for tab_def in self._TAB_DEFS:
            tab = QWidget()
            tab_layout = self._add_layout_to_tab(tab)
            self._sliders[tab_def.name] = {}
            for group_def in tab_def.slider_groups:
                self._sliders[tab_def.name][group_def.name] = self._create_slider_group(
                    tab_layout, group_def.name,
                    slider_config=group_def.slider_config,
                    default_values=group_def.default_values,
                    orientation=group_def.orientation,
                )
            tab_layout.addStretch()
            self.tab_widget.addTab(tab, tab_def.name)

        layout.addWidget(self.tab_widget)
        self.setLayout(layout)

    @property
    def color_config(self) -> ColorConfig:
        return self._get_tab_config("Color", ColorConfig)

    def set_color_config(self, config: ColorConfig) -> None:
        self._set_tab_config("Color", config)

    @property
    def geometry_config(self) -> GeometryConfig:
        return self._get_tab_config("Geometry", GeometryConfig)

    def set_geometry_config(self, config: GeometryConfig) -> None:
        self._set_tab_config("Geometry", config)

    @property
    def camera_config(self) -> CameraConfig:
        return self._get_tab_config("Camera Settings", CameraConfig)

    def set_camera_config(self, config: CameraConfig) -> None:
        self._set_tab_config("Camera Settings", config)

    def _get_tab_config(self, tab_name: str, config_class):
        tab_def = next(t for t in self._TAB_DEFS if t.name == tab_name)
        return config_class(**{
            g.name.lower().replace(" ", "_"): {
                lbl: self._sliders[tab_name][g.name][lbl].value()
                for lbl in g.slider_config
            }
            for g in tab_def.slider_groups
        })

    def _set_tab_config(self, tab_name: str, config) -> None:
        tab_def = next(t for t in self._TAB_DEFS if t.name == tab_name)
        for g in tab_def.slider_groups:
            group_values = getattr(config, g.name.lower().replace(" ", "_"))
            for lbl in g.slider_config:
                self._sliders[tab_name][g.name][lbl].setValue(group_values[lbl])

    def _add_layout_to_tab(self, tab: QWidget) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        tab.setLayout(layout)
        return layout

    def _create_slider_group(self, layout, title:str, slider_config:Dict[str,Tuple[Number, Number]], default_values:List[Number], orientation=Qt.Orientation.Vertical) -> Dict[str, QSlider]:
        """
        Add a separator and a group of sliders to the given layout.
        Args:
            layout: The layout to which the separator and sliders will be added.
            title: The title for the separator.
            slider_config: dict mapping slider labels to (min, max) tuples.
            default_value: Default value for sliders (int or dict of label->value)
            orientation: Qt.Orientation.Vertical or Qt.Orientation.Horizontal
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
        group_layout = QVBoxLayout()
        labels = list(slider_config.keys())
        sliders_layout = QHBoxLayout() if orientation == Qt.Orientation.Vertical else QVBoxLayout()
        sliders_layout.setSpacing(20)
        sliders: Dict[str, QSlider] = {}
        for lbl, default_value in zip(labels, default_values):
            slider_layout = QVBoxLayout()
            slider = QSlider(orientation)
            min_range, max_range = slider_config[lbl]
            slider.setMinimum(min_range)
            slider.setMaximum(max_range)
            slider.setValue(default_value)
            sliders[lbl] = slider
            slider_layout.addWidget(slider)
            label_widget = QLabel(lbl)
            label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            slider_layout.addWidget(label_widget, alignment=Qt.AlignmentFlag.AlignCenter)
            sliders_layout.addLayout(slider_layout)
        # Center the sliders
        centered_layout = QHBoxLayout() if orientation == Qt.Orientation.Vertical else QVBoxLayout()
        centered_layout.addStretch()
        centered_layout.addLayout(sliders_layout)
        centered_layout.addStretch()
        group_layout.addLayout(centered_layout)
        layout.addLayout(group_layout)
        return sliders