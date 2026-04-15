from typing import Dict, Iterable, Optional

from OpenGL import GL


_FRAMEBUFFER_STATUS_NAMES = {
    GL.GL_FRAMEBUFFER_UNDEFINED: "GL_FRAMEBUFFER_UNDEFINED",
    GL.GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT: "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT",
    GL.GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT",
    GL.GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER: "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER",
    GL.GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER: "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER",
    GL.GL_FRAMEBUFFER_UNSUPPORTED: "GL_FRAMEBUFFER_UNSUPPORTED",
    GL.GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE: "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE",
    GL.GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS: "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS",
}


class FramebufferAttachment:
    def __init__(self, attachment_point: int):
        self.attachment_point = attachment_point
        self.texture_id: Optional[int] = None

    def texture_target(self) -> int:
        raise NotImplementedError()

    def attach(self, framebuffer_id: int):
        raise NotImplementedError()

    def resize(self, width: int, height: int):
        raise NotImplementedError()

    def bind_for_sampling(self):
        if self.texture_id is None:
            raise RuntimeError("cannot sample from an uninitialized attachment")
        GL.glBindTexture(self.texture_target(), self.texture_id)

    def bind_layer(self, framebuffer_id: int, index: int):
        _ = framebuffer_id
        _ = index


class Color2DAttachment(FramebufferAttachment):
    def __init__(self):
        super().__init__(GL.GL_COLOR_ATTACHMENT0)

    def texture_target(self) -> int:
        return GL.GL_TEXTURE_2D

    def attach(self, framebuffer_id: int):
        self.texture_id = GL.glGenTextures(1)
        GL.glBindTexture(self.texture_target(), self.texture_id)
        GL.glTexParameteri(self.texture_target(), GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameteri(self.texture_target(), GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameteri(self.texture_target(), GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(self.texture_target(), GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            self.attachment_point,
            self.texture_target(),
            self.texture_id,
            0,
        )
        GL.glBindTexture(self.texture_target(), 0)

    def resize(self, width: int, height: int):
        GL.glBindTexture(self.texture_target(), self.texture_id)
        GL.glTexImage2D(self.texture_target(), 0, GL.GL_RGBA8, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glBindTexture(self.texture_target(), 0)


class Depth2DAttachment(FramebufferAttachment):
    def __init__(self):
        super().__init__(GL.GL_DEPTH_ATTACHMENT)

    def texture_target(self) -> int:
        return GL.GL_TEXTURE_2D

    def attach(self, framebuffer_id: int):
        self.texture_id = GL.glGenTextures(1)
        GL.glBindTexture(self.texture_target(), self.texture_id)
        GL.glTexParameteri(self.texture_target(), GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(self.texture_target(), GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(self.texture_target(), GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(self.texture_target(), GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        border_color = [1.0] * 4
        GL.glTexParameterfv(self.texture_target(), GL.GL_TEXTURE_BORDER_COLOR, border_color)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            self.attachment_point,
            self.texture_target(),
            self.texture_id,
            0,
        )
        GL.glBindTexture(self.texture_target(), 0)

    def resize(self, width: int, height: int):
        GL.glBindTexture(self.texture_target(), self.texture_id)
        GL.glTexImage2D(self.texture_target(), 0, GL.GL_DEPTH_COMPONENT, width, height, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)
        GL.glBindTexture(self.texture_target(), 0)


class DepthArrayAttachment(FramebufferAttachment):
    def __init__(self, layer_count: int):
        super().__init__(GL.GL_DEPTH_ATTACHMENT)
        self.layer_count = layer_count

    def texture_target(self) -> int:
        return GL.GL_TEXTURE_2D_ARRAY

    def attach(self, framebuffer_id: int):
        self.texture_id = GL.glGenTextures(1)
        GL.glBindTexture(self.texture_target(), self.texture_id)
        GL.glTexParameteri(self.texture_target(), GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(self.texture_target(), GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(self.texture_target(), GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(self.texture_target(), GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        border_color = [1.0] * 4
        GL.glTexParameterfv(self.texture_target(), GL.GL_TEXTURE_BORDER_COLOR, border_color)
        GL.glFramebufferTextureLayer(GL.GL_FRAMEBUFFER, self.attachment_point, self.texture_id, 0, 0)
        GL.glBindTexture(self.texture_target(), 0)

    def resize(self, width: int, height: int):
        GL.glBindTexture(self.texture_target(), self.texture_id)
        GL.glTexImage3D(
            self.texture_target(),
            0,
            GL.GL_DEPTH_COMPONENT,
            width,
            height,
            self.layer_count,
            0,
            GL.GL_DEPTH_COMPONENT,
            GL.GL_FLOAT,
            None,
        )
        GL.glBindTexture(self.texture_target(), 0)

    def bind_layer(self, framebuffer_id: int, index: int):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer_id)
        GL.glFramebufferTextureLayer(GL.GL_FRAMEBUFFER, self.attachment_point, self.texture_id, 0, index)


class CubeMapDepthArrayAttachment(FramebufferAttachment):
    def __init__(self, light_count: int):
        super().__init__(GL.GL_DEPTH_ATTACHMENT)
        self.light_count = light_count

    def texture_target(self) -> int:
        return GL.GL_TEXTURE_CUBE_MAP_ARRAY

    def attach(self, framebuffer_id: int):
        self.texture_id = GL.glGenTextures(1)
        GL.glBindTexture(self.texture_target(), self.texture_id)
        GL.glTexParameteri(self.texture_target(), GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(self.texture_target(), GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(self.texture_target(), GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(self.texture_target(), GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(self.texture_target(), GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)
        border_color = [1.0] * 4
        GL.glTexParameterfv(self.texture_target(), GL.GL_TEXTURE_BORDER_COLOR, border_color)
        GL.glFramebufferTexture(GL.GL_FRAMEBUFFER, self.attachment_point, self.texture_id, 0)
        GL.glBindTexture(self.texture_target(), 0)

    def resize(self, width: int, height: int):
        GL.glBindTexture(self.texture_target(), self.texture_id)
        GL.glTexImage3D(
            self.texture_target(),
            0,
            GL.GL_DEPTH_COMPONENT,
            width,
            height,
            6 * self.light_count,
            0,
            GL.GL_DEPTH_COMPONENT,
            GL.GL_FLOAT,
            None,
        )
        GL.glBindTexture(self.texture_target(), 0)


class ComposedFrameBuffer:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.glfboid = GL.glGenFramebuffers(1)
        self._attachments: Dict[str, FramebufferAttachment] = {}

    @property
    def hasColorBuffer(self) -> bool:
        return "color" in self._attachments

    @property
    def hasDepthBuffer(self) -> bool:
        depth_attachment = self._attachments.get("depth")
        return isinstance(depth_attachment, Depth2DAttachment)

    @property
    def hasMultiDepthBuffer(self) -> bool:
        depth_attachment = self._attachments.get("depth")
        return isinstance(depth_attachment, DepthArrayAttachment)

    @property
    def hasCubeMapDepthBuffer(self) -> bool:
        depth_attachment = self._attachments.get("depth")
        return isinstance(depth_attachment, CubeMapDepthArrayAttachment)

    @property
    def gltexid(self) -> Optional[int]:
        color_attachment = self._attachments.get("color")
        return color_attachment.texture_id if color_attachment else None

    @property
    def glrboid(self) -> Optional[int]:
        depth_attachment = self._attachments.get("depth")
        return depth_attachment.texture_id if depth_attachment else None

    def add_attachment(self, name: str, attachment: FramebufferAttachment):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.glfboid)
        attachment.attach(self.glfboid)
        self._attachments[name] = attachment
        self._set_draw_read_buffers()
        # Attachment storage is allocated in resize(); checking completeness here can
        # fail during initialization because textures have no image data yet.
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def _set_draw_read_buffers(self):
        if self.hasColorBuffer:
            GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        else:
            GL.glReadBuffer(GL.GL_NONE)
            GL.glDrawBuffer(GL.GL_NONE)

    def _assert_framebuffer_complete(self):
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            status_name = _FRAMEBUFFER_STATUS_NAMES.get(status, "UNKNOWN_FRAMEBUFFER_STATUS")
            raise RuntimeError(f"framebuffer incomplete with status {status} ({status_name})")

    def resize(self, value: Iterable[int]):
        self.width = value[0]
        self.height = value[1]
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.glfboid)
        for attachment in self._attachments.values():
            attachment.resize(self.width, self.height)
        self._set_draw_read_buffers()
        self._assert_framebuffer_complete()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def bind(self, index: int = 0):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.glfboid)
        depth_attachment = self._attachments.get("depth")
        if depth_attachment:
            depth_attachment.bind_layer(self.glfboid, index)

    def unbind(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def get_depth_texture_id(self) -> Optional[int]:
        depth_attachment = self._attachments.get("depth")
        return depth_attachment.texture_id if depth_attachment else None

    def get_depth_texture_target(self) -> Optional[int]:
        depth_attachment = self._attachments.get("depth")
        return depth_attachment.texture_target() if depth_attachment else None

    def get_color_texture_id(self) -> Optional[int]:
        color_attachment = self._attachments.get("color")
        return color_attachment.texture_id if color_attachment else None

    def colorToBuffer(self):
        if not self.hasColorBuffer:
            raise RuntimeError("colorToBuffer requires a color attachment")
        self.bind()
        pixels = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        self.unbind()
        return pixels

    def depthToBuffer(self):
        depth_attachment = self._attachments.get("depth")
        if depth_attachment is None:
            raise RuntimeError("depthToBuffer requires a depth attachment")
        if isinstance(depth_attachment, DepthArrayAttachment):
            return_value = []
            for i in range(depth_attachment.layer_count):
                self.bind(i)
                pixels = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
                return_value.append(pixels)
            self.unbind()
            return return_value
        self.bind()
        pixels = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
        self.unbind()
        return (pixels,)

    def bind_shadow_texture(self):
        depth_attachment = self._attachments.get("depth")
        if depth_attachment is None:
            raise NotImplementedError("binding shadow texture is not implemented for buffers without depth component")
        depth_attachment.bind_for_sampling()
