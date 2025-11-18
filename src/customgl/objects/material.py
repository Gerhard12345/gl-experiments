from pathlib import Path
from typing import List


class Material:
    def __init__(
        self,
        specularpower=32,
        texture_scales: List[float] = None,
        texturefilename: Path = None,
        texturefilename_normals: Path = None,
        texturefilename_ambient_occlusion: Path = None,
        texturefilename_specular: Path = None,
    ):
        self.texturefilename_specular = texturefilename_specular
        self.texturefilename_ambient_occlusion = texturefilename_ambient_occlusion
        self.texturefilename_normals = texturefilename_normals
        self.texturefilename = texturefilename
        self.specularpower = specularpower
        self.texture_scales = texture_scales if texture_scales is not None else [1, 1]


class WoodenCeiling(Material):
    def __init__(self, texturescales: List[float], specularpower: int = 128):
        super().__init__(
            texturefilename=Path(__file__).parent.parent / "textures" / "ceiling1" / "Wood_Ceiling_Coffers_003_basecolor.jpg",
            texturefilename_normals=Path(__file__).parent.parent / "textures" / "ceiling1" / "Wood_Ceiling_Coffers_003_NormalMap.png",
            texturefilename_ambient_occlusion=Path(__file__).parent.parent
            / "textures"
            / "ceiling1"
            / "Wood_Ceiling_Coffers_003_ambientOcclusion.jpg",
            texturefilename_specular=Path(__file__).parent.parent / "textures" / "ceiling1" / "Wood_Ceiling_Coffers_003_metallic.jpg",
            texture_scales=texturescales,
            specularpower=specularpower,
        )


class BrickWall1(Material):
    def __init__(self, texturescales: List[float] = None, specularpower: int = 32):
        super().__init__(
            texturefilename=Path(__file__).parent.parent / "textures" / "Brickwall2" / "Brick_Wall_019_basecolor.jpg",
            texturefilename_normals=Path(__file__).parent.parent / "textures" / "Brickwall2" / "Brick_Wall_019_normal.png",
            texturefilename_ambient_occlusion=Path(__file__).parent.parent / "textures" / "Brickwall2" / "Brick_Wall_019_ambientOcclusion.jpg",
            texturefilename_specular=Path(__file__).parent.parent / "textures" / "Brickwall2" / "Brick_Wall_019_SpecularMap.png",
            texture_scales=texturescales,
            specularpower=specularpower,
        )


class BrickWall2(Material):
    def __init__(self, texturescales: List[float] = None, specularpower: int = 32):
        super().__init__(
            texturefilename=Path(__file__).parent.parent / "textures" / "brickwall" / "brickwall.jpg",
            texturefilename_normals=Path(__file__).parent.parent / "textures" / "brickwall" / "brickwall_normal.png",
            texturefilename_ambient_occlusion=Path(__file__).parent.parent / "textures" / "brickwall" / "brickwall_ambient_occ.png",
            texturefilename_specular=Path(__file__).parent.parent / "textures" / "brickwall" / "brickwall_specular.png",
            texture_scales=texturescales,
            specularpower=specularpower,
        )


class Marble1(Material):
    def __init__(self, texturescales: List[float] = None, specularpower: int = 32):
        super().__init__(
            texturefilename=Path(__file__).parent.parent / "textures" / "marble1" / "Marble_White_006_basecolor.jpg",
            texturefilename_normals=Path(__file__).parent.parent / "textures" / "marble1" / "Marble_White_006_NormalMap.png",
            texturefilename_ambient_occlusion=Path(__file__).parent.parent / "textures" / "marble1" / "Marble_White_006_ambientOcclusion.jpg",
            texturefilename_specular=Path(__file__).parent.parent / "textures" / "marble1" / "Marble_White_006_SpecularMap.png",
            specularpower=specularpower,
            texture_scales=texturescales,
        )


class GoldFoil(Material):
    def __init__(self, texturescales: List[float] = None, specularpower: int = 128):
        super().__init__(
            texturefilename=Path(__file__).parent.parent / "textures" / "goldfoil" / "Metal_Gold_Foil_001_basecolor.png",
            texturefilename_normals=Path(__file__).parent.parent / "textures" / "goldfoil" / "Metal_Gold_Foil_001_NormalMap.png",
            texturefilename_ambient_occlusion=Path(__file__).parent.parent
            / "textures"
            / "goldfoil"
            / "Metal_Gold_Foil_001_ambientOcclusion.png",
            texturefilename_specular=Path(__file__).parent.parent / "textures" / "goldfoil" / "Metal_Gold_Foil_001_SpecularMap1.png",
            texture_scales=texturescales,
            specularpower=specularpower,
        )


class MuddyConcrete(Material):
    def __init__(self, texturescales: List[float] = None, specularpower: int = 128):
        super().__init__(
            texturefilename=Path(__file__).parent.parent / "textures" / "floor1" / "Concrete_Muddy_001_BaseColor.jpg",
            texturefilename_normals=Path(__file__).parent.parent / "textures" / "floor1" / "Concrete_Muddy_001_NormalMap.png",
            texturefilename_ambient_occlusion=Path(__file__).parent.parent / "textures" / "floor1" / "Concrete_Muddy_001_AmbientOcclusion.jpg",
            texturefilename_specular=Path(__file__).parent.parent / "textures" / "floor1" / "Concrete_Muddy_001_SpecularMap.png",
            texture_scales=texturescales,
            specularpower=specularpower,
        )


class TerraCottaTiles(Material):
    def __init__(self, texturescales: List[float] = None, specularpower: int = 128):
        super().__init__(
            texturefilename=Path(__file__).parent.parent / "textures" / "TerraCotta1" / "Terracotta_Floor_Tiles_004_basecolor.png",
            texturefilename_normals=Path(__file__).parent.parent / "textures" / "TerraCotta1" / "Terracotta_Floor_Tiles_004_NormalMap.png",
            texturefilename_ambient_occlusion=Path(__file__).parent.parent
            / "textures"
            / "TerraCotta1"
            / "Terracotta_Floor_Tiles_004_ambientOcclusion.png",
            texturefilename_specular=Path(__file__).parent.parent / "textures" / "TerraCotta1" / "Terracotta_Floor_Tiles_004_SpecularMap.png",
            texture_scales=texturescales,
            specularpower=specularpower,
        )


class Wood1(Material):
    def __init__(self, texturescales: List[float] = None, specularpower: int = 128):
        super().__init__(
            texturefilename=Path(__file__).parent.parent / "textures" / "Wood1" / "Wood_025_basecolor.jpg",
            texturefilename_normals=Path(__file__).parent.parent / "textures" / "Wood1" / "Wood_025_NormalMap.png",
            texturefilename_ambient_occlusion=Path(__file__).parent.parent / "textures" / "Wood1" / "Wood_025_ambientOcclusion.jpg",
            texturefilename_specular=Path(__file__).parent.parent / "textures" / "Wood1" / "Wood_025_SpecularMap.png",
            texture_scales=texturescales,
            specularpower=specularpower,
        )


class MetalPanel1(Material):
    def __init__(self, texturescales: List[float] = None, specularpower: int = 128):
        super().__init__(
            texturefilename=Path(__file__).parent.parent / "textures" / "MetalPanel1" / "Sci_fi_Metal_Panel_007_basecolor.png",
            texturefilename_normals=Path(__file__).parent.parent / "textures" / "MetalPanel1" / "Sci_fi_Metal_Panel_007_NormalMap.png",
            texturefilename_ambient_occlusion=Path(__file__).parent.parent
            / "textures"
            / "MetalPanel1"
            / "Sci_fi_Metal_Panel_007_ambientOcclusion.png",
            texturefilename_specular=Path(__file__).parent.parent / "textures" / "MetalPanel1" / "Sci_fi_Metal_Panel_007_SpecularMap.png",
            texture_scales=texturescales,
            specularpower=specularpower,
        )


class WhiteBricks(Material):
    def __init__(self, texturescales: List[float] = None, specularpower: int = 128):
        super().__init__(
            texturefilename=Path(__file__).parent.parent / "textures" / "painted-white-bricks-bl" / "painted-white-bricks_albedo.png",
            texturefilename_normals=Path(__file__).parent.parent / "textures" / "painted-white-bricks-bl" / "painted-white-bricks_normal-dx.png",
            texturefilename_ambient_occlusion=Path(__file__).parent.parent
            / "textures"
            / "painted-white-bricks-bl"
            / "painted-white-bricks_ao.png",
            texturefilename_specular=Path(__file__).parent.parent / "textures" / "painted-white-bricks-bl" / "painted-white-bricks_metallic.png",
            texture_scales=texturescales,
            specularpower=specularpower,
        )


class WornMetal(Material):
    def __init__(self, texturescales: List[float] = None, specularpower: int = 128):
        super().__init__(
            texturefilename=Path(__file__).parent.parent / "textures" / "worn-painted-metal" / "worn-painted-metal_albedo.png",
            texturefilename_normals=Path(__file__).parent.parent / "textures" / "worn-painted-metal" / "worn-painted-metal_normal-dx.png",
            texturefilename_ambient_occlusion=Path(__file__).parent.parent / "textures" / "worn-painted-metal" / "worn-painted-metal_ao.png",
            texturefilename_specular=Path(__file__).parent.parent / "textures" / "worn-painted-metal" / "worn-painted-metal_roughness.png",
            texture_scales=texturescales,
            specularpower=specularpower,
        )
