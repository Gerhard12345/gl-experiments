from .framebuffer_composition import Color2DAttachment
from .framebuffer_composition import ComposedFrameBuffer
from .framebuffer_composition import CubeMapDepthArrayAttachment
from .framebuffer_composition import Depth2DAttachment
from .framebuffer_composition import DepthArrayAttachment
from .framebuffer_composition import FramebufferAttachment


class CustomFrameBuffer(ComposedFrameBuffer):
    def __init__(self):
        super().__init__()

    @classmethod
    def with_rgb_and_depth(cls):
        framebuffer = cls()
        framebuffer.add_color_buffer()
        framebuffer.add_depth_buffer()
        return framebuffer

    @classmethod
    def with_depth_only(cls):
        framebuffer = cls()
        framebuffer.add_depth_buffer()
        return framebuffer

    @classmethod
    def with_multi_depth(cls, n_layers: int):
        framebuffer = cls()
        framebuffer.add_multi_depth_buffer(layer_count=n_layers)
        return framebuffer

    @classmethod
    def with_cubemap_depth(cls, n_lights: int):
        framebuffer = cls()
        framebuffer.add_cubemap_depth_buffer(light_count=n_lights)
        return framebuffer

    def add_color_buffer(self):
        self.add_attachment("color", Color2DAttachment())

    def add_depth_buffer(self):
        self.add_attachment("depth", Depth2DAttachment())

    def add_cubemap_depth_buffer(self, light_count: int):
        self.add_attachment("depth", CubeMapDepthArrayAttachment(light_count=light_count))

    def add_multi_depth_buffer(self, layer_count: int):
        self.add_attachment("depth", DepthArrayAttachment(layer_count=layer_count))


__all__ = [
    "FramebufferAttachment",
    "Color2DAttachment",
    "Depth2DAttachment",
    "DepthArrayAttachment",
    "CubeMapDepthArrayAttachment",
    "ComposedFrameBuffer",
    "CustomFrameBuffer",
]
