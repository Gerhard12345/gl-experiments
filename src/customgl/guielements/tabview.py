from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget, QSlider,
    QLabel, QPushButton, QFrame, QPlainTextEdit, QSplitter, QSpacerItem,
    QSizePolicy, QSplitterHandle
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPainter, QColor, QPalette


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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create the tab widget
        self.tab_widget = QTabWidget()

        # Color Tab
        color_tab = QWidget()
        color_layout = QVBoxLayout()
        color_layout.setContentsMargins(5, 5, 5, 5)
        color_layout.setSpacing(5)

        # Ambient group
        color_layout.addWidget(self._create_separator("Ambient"))
        color_layout.addLayout(self._create_slider_group())

        # Diffuse group
        color_layout.addWidget(self._create_separator("Diffuse"))
        color_layout.addLayout(self._create_slider_group())

        # Specular group
        color_layout.addWidget(self._create_separator("Specular"))
        color_layout.addLayout(self._create_slider_group())

        color_layout.addStretch()
        color_tab.setLayout(color_layout)
        self.tab_widget.addTab(color_tab, "Color")

        # Geometry Tab
        geometry_tab = QWidget()
        geometry_layout = QVBoxLayout()
        geometry_layout.setContentsMargins(5, 5, 5, 5)
        geometry_layout.setSpacing(5)

        # Ambient direction group
        geometry_layout.addWidget(self._create_separator("Ambient direction"))
        geometry_layout.addLayout(
            self._create_slider_group(
                slider_config={"X": (0, 100), "Y": (0, 100), "Z": (0, 100)}
            )
        )

        # Point light position group
        geometry_layout.addWidget(self._create_separator("Point light position"))
        geometry_layout.addLayout(
            self._create_slider_group(
                slider_config={"X": (0, 100), "Y": (0, 100), "Z": (0, 100)}
            )
        )

        geometry_layout.addStretch()
        geometry_tab.setLayout(geometry_layout)
        self.tab_widget.addTab(geometry_tab, "Geometry")

        # Camera Settings Tab
        camera_tab = QWidget()
        camera_layout = QVBoxLayout()
        camera_layout.setContentsMargins(5, 5, 5, 5)
        camera_layout.setSpacing(5)

        # Field of View slider group
        camera_layout.addWidget(self._create_separator("Field of View"))
        camera_layout.addLayout(self._create_slider_group({"FOV": (1, 179)}, default_value=60, orientation=Qt.Orientation.Horizontal))
        camera_layout.addStretch()
        camera_tab.setLayout(camera_layout)
        self.tab_widget.addTab(camera_tab, "Camera Settings")

        layout.addWidget(self.tab_widget)
        self.setLayout(layout)

    def _create_slider_group(self, slider_config=None, min_val=0, max_val=255, default_value=None, orientation=Qt.Orientation.Vertical):
        """Create a layout with any number of sliders.
        Args:
            slider_config: dict mapping slider labels to (min, max) tuples. If None, defaults to RGB (R, G, B).
            min_val: Default minimum value (used if slider_config is None)
            max_val: Default maximum value (used if slider_config is None)
            default_value: Default value for sliders (int or dict of label->value)
            orientation: Qt.Orientation.Vertical or Qt.Orientation.Horizontal
        """
        group_layout = QVBoxLayout()
        if slider_config is None:
            labels = ["R", "G", "B"]
            slider_config = {lbl: (min_val, max_val) for lbl in labels}
        else:
            labels = list(slider_config.keys())

        sliders_layout = QHBoxLayout() if orientation == Qt.Orientation.Vertical else QVBoxLayout()
        sliders_layout.setSpacing(20)

        for lbl in labels:
            slider_layout = QVBoxLayout()
            slider = QSlider(orientation)
            min_range, max_range = slider_config[lbl]
            slider.setMinimum(min_range)
            slider.setMaximum(max_range)
            if isinstance(default_value, dict):
                slider.setValue(default_value.get(lbl, min_range))
            elif default_value is not None:
                slider.setValue(default_value)
            else:
                slider.setValue((min_range + max_range) // 2)
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
        return group_layout

    def _create_separator(self, title):
        """Create a separator with title text centered in a line."""
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
