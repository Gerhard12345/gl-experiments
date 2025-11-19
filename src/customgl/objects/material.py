from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Material:
    specularpower: int = 32
    texture_scales: List[float] = field(default_factory=lambda: [1, 1])
    texturefilename: Path = None
    texturefilename_normals: Path = None
    texturefilename_ambient_occlusion: Path = None
    texturefilename_specular: Path = None


@dataclass
class WoodenCeiling(Material):
    texturefilename: Path = Path(__file__).parent.parent / "textures" / "ceiling1" / "Wood_Ceiling_Coffers_003_basecolor.jpg"
    texturefilename_normals: Path = Path(__file__).parent.parent / "textures" / "ceiling1" / "Wood_Ceiling_Coffers_003_NormalMap.png"
    texturefilename_ambient_occlusion: Path = (
        Path(__file__).parent.parent / "textures" / "ceiling1" / "Wood_Ceiling_Coffers_003_ambientOcclusion.jpg"
    )
    texturefilename_specular: Path = Path(__file__).parent.parent / "textures" / "ceiling1" / "Wood_Ceiling_Coffers_003_metallic.jpg"
    specularpower: int = 128


@dataclass
class BrickWall1(Material):
    texturefilename: Path = Path(__file__).parent.parent / "textures" / "Brickwall2" / "Brick_Wall_019_basecolor.jpg"
    texturefilename_normals: Path = Path(__file__).parent.parent / "textures" / "Brickwall2" / "Brick_Wall_019_normal.png"
    texturefilename_ambient_occlusion: Path = Path(__file__).parent.parent / "textures" / "Brickwall2" / "Brick_Wall_019_ambientOcclusion.jpg"
    texturefilename_specular: Path = Path(__file__).parent.parent / "textures" / "Brickwall2" / "Brick_Wall_019_SpecularMap.png"
    specularpower: int = 32


@dataclass
class BrickWall2(Material):
    texturefilename: Path = Path(__file__).parent.parent / "textures" / "brickwall" / "brickwall.jpg"
    texturefilename_normals: Path = Path(__file__).parent.parent / "textures" / "brickwall" / "brickwall_normal.png"
    texturefilename_ambient_occlusion: Path = Path(__file__).parent.parent / "textures" / "brickwall" / "brickwall_ambient_occ.png"
    texturefilename_specular: Path = Path(__file__).parent.parent / "textures" / "brickwall" / "brickwall_specular.png"
    specularpower: int = 32


@dataclass
class Marble1(Material):
    texturefilename: Path = Path(__file__).parent.parent / "textures" / "marble1" / "Marble_White_006_basecolor.jpg"
    texturefilename_normals: Path = Path(__file__).parent.parent / "textures" / "marble1" / "Marble_White_006_NormalMap.png"
    texturefilename_ambient_occlusion: Path = Path(__file__).parent.parent / "textures" / "marble1" / "Marble_White_006_ambientOcclusion.jpg"
    texturefilename_specular: Path = Path(__file__).parent.parent / "textures" / "marble1" / "Marble_White_006_SpecularMap.png"
    texture_scales = 32


@dataclass
class GoldFoil(Material):
    texturefilename: Path = Path(__file__).parent.parent / "textures" / "goldfoil" / "Metal_Gold_Foil_001_basecolor.png"
    texturefilename_normals: Path = Path(__file__).parent.parent / "textures" / "goldfoil" / "Metal_Gold_Foil_001_NormalMap.png"
    texturefilename_ambient_occlusion: Path = Path(__file__).parent.parent / "textures" / "goldfoil" / "Metal_Gold_Foil_001_ambientOcclusion.png"
    texturefilename_specular: Path = Path(__file__).parent.parent / "textures" / "goldfoil" / "Metal_Gold_Foil_001_SpecularMap1.png"
    specularpower: int = 128


@dataclass
class MuddyConcrete(Material):
    texturefilename: Path = Path(__file__).parent.parent / "textures" / "floor1" / "Concrete_Muddy_001_BaseColor.jpg"
    texturefilename_normals: Path = Path(__file__).parent.parent / "textures" / "floor1" / "Concrete_Muddy_001_NormalMap.png"
    texturefilename_ambient_occlusion: Path = Path(__file__).parent.parent / "textures" / "floor1" / "Concrete_Muddy_001_AmbientOcclusion.jpg"
    texturefilename_specular: Path = Path(__file__).parent.parent / "textures" / "floor1" / "Concrete_Muddy_001_SpecularMap.png"
    specularpower: int = 128


@dataclass
class TerraCottaTiles(Material):
    texturefilename: Path = Path(__file__).parent.parent / "textures" / "TerraCotta1" / "Terracotta_Floor_Tiles_004_basecolor.png"
    texturefilename_normals: Path = Path(__file__).parent.parent / "textures" / "TerraCotta1" / "Terracotta_Floor_Tiles_004_NormalMap.png"
    texturefilename_ambient_occlusion: Path = (
        Path(__file__).parent.parent / "textures" / "TerraCotta1" / "Terracotta_Floor_Tiles_004_ambientOcclusion.png"
    )
    texturefilename_specular: Path = Path(__file__).parent.parent / "textures" / "TerraCotta1" / "Terracotta_Floor_Tiles_004_SpecularMap.png"
    specularpower: int = 128


@dataclass
class Wood1(Material):
    texturefilename: Path = Path(__file__).parent.parent / "textures" / "Wood1" / "Wood_025_basecolor.jpg"
    texturefilename_normals: Path = Path(__file__).parent.parent / "textures" / "Wood1" / "Wood_025_NormalMap.png"
    texturefilename_ambient_occlusion: Path = Path(__file__).parent.parent / "textures" / "Wood1" / "Wood_025_ambientOcclusion.jpg"
    texturefilename_specular: Path = Path(__file__).parent.parent / "textures" / "Wood1" / "Wood_025_SpecularMap.png"
    specularpower: int = 128


@dataclass
class MetalPanel1(Material):
    texturefilename: Path = Path(__file__).parent.parent / "textures" / "MetalPanel1" / "Sci_fi_Metal_Panel_007_basecolor.png"
    texturefilename_normals: Path = Path(__file__).parent.parent / "textures" / "MetalPanel1" / "Sci_fi_Metal_Panel_007_NormalMap.png"
    texturefilename_ambient_occlusion: Path = (
        Path(__file__).parent.parent / "textures" / "MetalPanel1" / "Sci_fi_Metal_Panel_007_ambientOcclusion.png"
    )
    texturefilename_specular: Path = Path(__file__).parent.parent / "textures" / "MetalPanel1" / "Sci_fi_Metal_Panel_007_SpecularMap.png"
    specularpower: int = 128


@dataclass
class WhiteBricks(Material):
    # taken from https://freepbr.com/product/painted-white-bricks-pbr/
    texturefilename: Path = Path(__file__).parent.parent / "textures" / "painted-white-bricks-bl" / "painted-white-bricks_albedo.png"
    texturefilename_normals: Path = Path(__file__).parent.parent / "textures" / "painted-white-bricks-bl" / "painted-white-bricks_normal-ogl.png"
    texturefilename_ambient_occlusion: Path = (
        Path(__file__).parent.parent / "textures" / "painted-white-bricks-bl" / "painted-white-bricks_ao.png"
    )
    texturefilename_specular: Path = Path(__file__).parent.parent / "textures" / "painted-white-bricks-bl" / "painted-white-bricks_metallic.png"
    specularpower: int = 128


@dataclass
class WornMetal(Material):
    # taken from https://freepbr.com/product/worn-painted-metal/
    texturefilename: Path = Path(__file__).parent.parent / "textures" / "worn-painted-metal" / "worn-painted-metal_albedo.png"
    texturefilename_normals: Path = Path(__file__).parent.parent / "textures" / "worn-painted-metal" / "worn-painted-metal_normal-ogl.png"
    texturefilename_ambient_occlusion: Path = Path(__file__).parent.parent / "textures" / "worn-painted-metal" / "worn-painted-metal_ao.png"
    texturefilename_specular: Path = Path(__file__).parent.parent / "textures" / "worn-painted-metal" / "worn-painted-metal_roughness.png"
    specularpower: int = 128
