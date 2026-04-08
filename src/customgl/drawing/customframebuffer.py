from .framebuffer_composition import Color2DAttachment
from .framebuffer_composition import ComposedFrameBuffer
from .framebuffer_composition import CubeMapDepthArrayAttachment
from .framebuffer_composition import Depth2DAttachment
from .framebuffer_composition import DepthArrayAttachment
from .framebuffer_composition import FramebufferAttachment


class CustomFrameBuffer(ComposedFrameBuffer):
    def __init__(self, n_lights: int):
        super().__init__(n_lights=n_lights)

    @classmethod
    def with_rgb_and_depth(cls, n_lights: int):
        framebuffer = cls(n_lights=n_lights)
        framebuffer.addColorBuffer()
        framebuffer.addDepthBuffer()
        return framebuffer

    @classmethod
    def with_depth_only(cls, n_lights: int):
        framebuffer = cls(n_lights=n_lights)
        framebuffer.addDepthBuffer()
        return framebuffer

    @classmethod
    def with_multi_depth(cls, n_lights: int):
        framebuffer = cls(n_lights=n_lights)
        framebuffer.addMultiDepthBuffer()
        return framebuffer

    @classmethod
    def with_cubemap_depth(cls, n_lights: int):
        framebuffer = cls(n_lights=n_lights)
        framebuffer.addCubeMapDepthBuffer()
        return framebuffer

    def addColorBuffer(self):
        self.add_attachment("color", Color2DAttachment())

    def addDepthBuffer(self):
        self.add_attachment("depth", Depth2DAttachment())

    def addCubeMapDepthBuffer(self):
        self.add_attachment("depth", CubeMapDepthArrayAttachment(light_count=self.n_lights))

    def addMultiDepthBuffer(self):
        self.add_attachment("depth", DepthArrayAttachment(layer_count=self.n_lights))


__all__ = [
    "FramebufferAttachment",
    "Color2DAttachment",
    "Depth2DAttachment",
    "DepthArrayAttachment",
    "CubeMapDepthArrayAttachment",
    "ComposedFrameBuffer",
    "CustomFrameBuffer",
]
