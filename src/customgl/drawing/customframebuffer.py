from typing import Iterable
from OpenGL.GL import *


class CustomFrameBuffer:
    def __init__(self, n_lights: int):
        self.n_lights = n_lights
        self.width = 0
        self.height = 0
        self.glfboid = glGenFramebuffers(1)
        self.hasDepthBuffer = False
        self.hasMultiDepthBuffer = False
        self.hasColorBuffer = False
        self.hasCubeMapDepthBuffer = False
        self.gltexid: int = None
        self.glrboid: int = None

    def addColorBuffer(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.glfboid)
        self.gltexid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.gltexid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        # Use the currently bound texture as color attachment
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.gltexid, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        self.hasColorBuffer = True

    def addDepthBuffer(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.glfboid)
        self.glrboid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.glrboid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        border_color = [1.0] * 4
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.glrboid, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self.hasDepthBuffer = True

    def addCubeMapDepthBuffer(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.glfboid)
        self.glrboid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, self.glrboid)
        glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        border_color = [1.0] * 8
        glTexParameterfv(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_BORDER_COLOR, border_color)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, self.glrboid, 0)
        glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self.hasCubeMapDepthBuffer = True

    def addMultiDepthBuffer(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.glfboid)
        self.glrboid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.glrboid)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT)
        border_color = [1.0] * 6
        glTexParameterfv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BORDER_COLOR, border_color)
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, self.glrboid, 0, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
        self.hasMultiDepthBuffer = True

    def resize(self, value: Iterable[int]):
        self.width = value[0]
        self.height = value[1]
        print(f"setting framebuffer size to w = {value[0]}, h = {value[1]}")
        print(f"Framebuffer texture id = {self.glrboid}")
        glBindFramebuffer(GL_FRAMEBUFFER, self.glfboid)
        if self.hasColorBuffer:
            glBindTexture(GL_TEXTURE_2D, self.gltexid)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            glBindTexture(GL_TEXTURE_2D, 0)
        else:
            glReadBuffer(GL_NONE)
            glDrawBuffer(GL_NONE)
        if self.hasDepthBuffer:
            glBindTexture(GL_TEXTURE_2D, self.glrboid)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, self.width, self.height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
            glBindTexture(GL_TEXTURE_2D, 0)
        if self.hasMultiDepthBuffer:
            glBindTexture(GL_TEXTURE_2D_ARRAY, self.glrboid)
            glTexImage3D(
                GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT, self.width, self.height, self.n_lights, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None
            )
            glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
        if self.hasCubeMapDepthBuffer:
            glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, self.glrboid)
            glTexImage3D(
                GL_TEXTURE_CUBE_MAP_ARRAY,
                0,
                GL_DEPTH_COMPONENT,
                self.width,
                self.height,
                6 * self.n_lights,
                0,
                GL_DEPTH_COMPONENT,
                GL_FLOAT,
                None,
            )
            glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def bind(self, index=0):
        glBindFramebuffer(GL_FRAMEBUFFER, self.glfboid)
        if self.hasMultiDepthBuffer:
            glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, self.glrboid, 0, index)
        # if self.hasCubeMapDepthBuffer:
        #    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, self.glrboid, 0, 0)

    def unbind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def colorToBuffer(self):
        self.bind()
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE)
        self.unbind()
        return pixels

    def depthToBuffer(self):
        return_value = []
        if self.hasMultiDepthBuffer:
            for i in range(self.n_lights):
                self.bind(i)
                pixels = glReadPixels(0, 0, self.width, self.height, GL_DEPTH_COMPONENT, GL_FLOAT)
                return_value.append(pixels)
        else:
            return_value = (glReadPixels(0, 0, self.width, self.height, GL_DEPTH_COMPONENT, GL_FLOAT),)
        self.unbind()
        return return_value

    def bind_shadow_texture(self):
        glActiveTexture(GL_TEXTURE0)
        if self.hasMultiDepthBuffer:
            glBindTexture(GL_TEXTURE_2D_ARRAY, self.glrboid)
        elif self.hasDepthBuffer:
            glBindTexture(GL_TEXTURE_2D, self.glrboid)
        elif self.hasCubeMapDepthBuffer:
            glActiveTexture(GL_TEXTURE5)
            glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, self.glrboid)
        else:
            raise NotImplementedError("binding shadow texture is not implemented fot buffers without depth component")
