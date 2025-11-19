from typing import Iterable
from OpenGL import GL


class CustomFrameBuffer:
    def __init__(self, n_lights: int):
        self.n_lights = n_lights
        self.width = 0
        self.height = 0
        self.glfboid = GL.glGenFramebuffers(1)
        self.hasDepthBuffer = False
        self.hasMultiDepthBuffer = False
        self.hasColorBuffer = False
        self.hasCubeMapDepthBuffer = False
        self.gltexid: int = None
        self.glrboid: int = None

    def addColorBuffer(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.glfboid)
        self.gltexid = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.gltexid)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        # Use the currently bound texture as color attachment
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.gltexid, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        self.hasColorBuffer = True

    def addDepthBuffer(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.glfboid)
        self.glrboid = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.glrboid)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        border_color = [1.0] * 4
        GL.glTexParameterfv(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BORDER_COLOR, border_color)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, self.glrboid, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        self.hasDepthBuffer = True

    def addCubeMapDepthBuffer(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.glfboid)
        self.glrboid = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP_ARRAY, self.glrboid)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP_ARRAY, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP_ARRAY, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP_ARRAY, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP_ARRAY, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP_ARRAY, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)
        border_color = [1.0] * 8
        GL.glTexParameterfv(GL.GL_TEXTURE_CUBE_MAP_ARRAY, GL.GL_TEXTURE_BORDER_COLOR, border_color)
        GL.glFramebufferTexture(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, self.glrboid, 0)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP_ARRAY, 0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        self.hasCubeMapDepthBuffer = True

    def addMultiDepthBuffer(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.glfboid)
        self.glrboid = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self.glrboid)
        GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        border_color = [1.0] * 6
        GL.glTexParameterfv(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_BORDER_COLOR, border_color)
        GL.glFramebufferTextureLayer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, self.glrboid, 0, 0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, 0)
        self.hasMultiDepthBuffer = True

    def resize(self, value: Iterable[int]):
        self.width = value[0]
        self.height = value[1]
        print(f"setting framebuffer size to w = {value[0]}, h = {value[1]}")
        print(f"Framebuffer texture id = {self.glrboid}")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.glfboid)
        if self.hasColorBuffer:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.gltexid)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, self.width, self.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        else:
            GL.glReadBuffer(GL.GL_NONE)
            GL.glDrawBuffer(GL.GL_NONE)
        if self.hasDepthBuffer:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.glrboid)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT, self.width, self.height, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        if self.hasMultiDepthBuffer:
            GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self.glrboid)
            GL.glTexImage3D(
                GL.GL_TEXTURE_2D_ARRAY,
                0,
                GL.GL_DEPTH_COMPONENT,
                self.width,
                self.height,
                self.n_lights,
                0,
                GL.GL_DEPTH_COMPONENT,
                GL.GL_FLOAT,
                None,
            )
            GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, 0)
        if self.hasCubeMapDepthBuffer:
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP_ARRAY, self.glrboid)
            GL.glTexImage3D(
                GL.GL_TEXTURE_CUBE_MAP_ARRAY,
                0,
                GL.GL_DEPTH_COMPONENT,
                self.width,
                self.height,
                6 * self.n_lights,
                0,
                GL.GL_DEPTH_COMPONENT,
                GL.GL_FLOAT,
                None,
            )
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, 0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def bind(self, index=0):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.glfboid)
        if self.hasMultiDepthBuffer:
            GL.glFramebufferTextureLayer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, self.glrboid, 0, index)

    def unbind(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def colorToBuffer(self):
        self.bind()
        pixels = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        self.unbind()
        return pixels

    def depthToBuffer(self):
        return_value = []
        if self.hasMultiDepthBuffer:
            for i in range(self.n_lights):
                self.bind(i)
                pixels = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
                return_value.append(pixels)
        else:
            return_value = (GL.glReadPixels(0, 0, self.width, self.height, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT),)
        self.unbind()
        return return_value

    def bind_shadow_texture(self):
        GL.glActiveTexture(GL.GL_TEXTURE0)
        if self.hasMultiDepthBuffer:
            GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self.glrboid)
        elif self.hasDepthBuffer:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.glrboid)
        elif self.hasCubeMapDepthBuffer:
            GL.glActiveTexture(GL.GL_TEXTURE5)
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP_ARRAY, self.glrboid)
        else:
            raise NotImplementedError("binding shadow texture is not implemented fot buffers without depth component")
