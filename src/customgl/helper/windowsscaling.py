import ctypes
from enum import IntEnum


class WidthIdentifers(IntEnum):
    VIRTUAL_WIDTH = 8
    PHYSICAL_WIDTH = 118

def get_windows_scaling_factor():
    dc = ctypes.windll.user32.GetDC(0)
    virtual_width = ctypes.windll.gdi32.GetDeviceCaps(dc, WidthIdentifers.VIRTUAL_WIDTH)
    physical_width = ctypes.windll.gdi32.GetDeviceCaps(dc, WidthIdentifers.PHYSICAL_WIDTH)
    scale = physical_width / virtual_width
    return scale