from pathlib import Path
from imageio import imread
import numpy as np
from numpy.typing import NDArray
import OpenGL.GL as GL
from ..objects.material import Material


class Texture:
    def __init__(
        self,
        filename: Path = None,
        filename_normal: Path = None,
        filename_ambient_occlusion: Path = None,
        filename_specular: Path = None,
    ):
        self.show_diffuse_maps = True
        self.show_normal_maps = True
        self.show_ambient_occlusion_maps = True
        self.show_specular_maps = True
        self.gltexid = self._generate_texture(filename=filename, default_value=np.array([[[128, 128, 128, 255]]]))
        self.glnormalid = self._generate_texture(filename=filename_normal, default_value=np.array([[[128, 128, 255, 255]]]))
        self.glambient_occlusion = self._generate_texture(filename=filename_ambient_occlusion, default_value=np.array([[[128, 128, 128, 255]]]))
        self.glspecular = self._generate_texture(filename=filename_specular, default_value=np.array([[[128, 128, 128, 255]]]))
        self.gltexid_plain = self._generate_texture(filename=None, default_value=np.array([[[128, 128, 128, 255]]]))
        self.glnormalid_plain = self._generate_texture(filename=None, default_value=np.array([[[128, 128, 255, 255]]]))
        self.glambient_occlusion_plain = self._generate_texture(filename=None, default_value=np.array([[[128, 128, 128, 255]]]))
        self.glspecular_plain = self._generate_texture(filename=None, default_value=np.array([[[128, 128, 128, 255]]]))

    def _generate_texture(self, filename: Path, default_value: NDArray[np.uint8]) -> int:
        local_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, local_texture)
        if not filename:
            im = default_value.astype(np.uint8)
        else:
            im = imread(filename)
            im = np.flip(im, axis=0)
        mode = GL.GL_RGBA
        if im.ndim == 2:
            mode = GL.GL_RED
        elif im.ndim == 3:
            if im.shape[2] == 3:
                mode = GL.GL_RGB
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, im.shape[1], im.shape[0], 0, mode, GL.GL_UNSIGNED_BYTE, im.tobytes())
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return local_texture

    def bind(self):
        texture_units = [GL.GL_TEXTURE1, GL.GL_TEXTURE2, GL.GL_TEXTURE3, GL.GL_TEXTURE4]
        tids = [self.gltexid, self.glnormalid, self.glambient_occlusion, self.glspecular]
        tids_plain = [self.gltexid_plain, self.glnormalid_plain, self.glambient_occlusion_plain, self.glspecular_plain]
        indicators = [self.show_diffuse_maps, self.show_normal_maps, self.show_ambient_occlusion_maps, self.show_specular_maps]
        for tu, tid, tid_plain, indicator in zip(texture_units, tids, tids_plain, indicators):
            GL.glActiveTexture(tu)
            texture_id = tid if indicator else tid_plain
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)

    def unbind(self):
        texture_units = [GL.GL_TEXTURE1, GL.GL_TEXTURE2, GL.GL_TEXTURE3, GL.GL_TEXTURE4]
        for tu in texture_units:
            GL.glActiveTexture(tu)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def toggle_detailed_diffuse_maps(self):
        self.show_diffuse_maps = not self.show_diffuse_maps

    def toggle_detailed_normal_maps(self):
        self.show_normal_maps = not self.show_normal_maps

    def toggle_detailed_ambient_occlusion_maps(self):
        self.show_ambient_occlusion_maps = not self.show_ambient_occlusion_maps

    def toggle_detailed_specular_maps(self):
        self.show_specular_maps = not self.show_specular_maps


class GLMaterial:
    def __init__(self, material: Material):
        self.texture = Texture(
            filename=material.texturefilename,
            filename_normal=material.texturefilename_normals,
            filename_ambient_occlusion=material.texturefilename_ambient_occlusion,
            filename_specular=material.texturefilename_specular,
        )
        self.specularpower = material.specularpower
        self.texturescales = material.texture_scales

    def setMaterial(self):
        program = GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM)
        u_specularpower = GL.glGetUniformLocation(program, bytes("u_material.specular_power", "utf-8"))
        GL.glUniform1f(u_specularpower, self.specularpower)
        u_int = GL.glGetUniformLocation(program, bytes("u_material.diffuse", "utf-8"))
        GL.glUniform1i(u_int, 1)
        u_int = GL.glGetUniformLocation(program, bytes("u_material.normal", "utf-8"))
        GL.glUniform1i(u_int, 2)
        u_int = GL.glGetUniformLocation(program, bytes("u_material.ambient_occlusion", "utf-8"))
        GL.glUniform1i(u_int, 3)
        u_int = GL.glGetUniformLocation(program, bytes("u_material.specular", "utf-8"))
        GL.glUniform1i(u_int, 4)
        u_texturescale = GL.glGetUniformLocation(program, bytes("u_texturescale", "utf-8"))
        GL.glUniform2f(u_texturescale, *self.texturescales)

    def __enter__(self):
        self.texture.bind()
        self.setMaterial()

    def __exit__(self, exc_type, exc_value, traceback):
        self.texture.unbind()
