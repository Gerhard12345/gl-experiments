from pathlib import Path
from src.customgl.objects.material import Material

class TestMaterial:
    def setup_method(self):
        print("Setup")
    
    def teardown_method(self):
        print("Teardown")

    def test_constructor(self):
        specularpower="str"
        texture_scales=[2,4,3]
        texturefilename=Path("texturefilename") 
        texturefilename_normals=Path("texturefilename_normals")
        texturefilename_ambient_occlusion=Path("texturefilename_ambient_occlusion")
        texturefilename_specular=Path("texturefilename_specular")
        material = Material(specularpower=specularpower,
                                 texture_scales=texture_scales,
                                 texturefilename=texturefilename,
                                 texturefilename_normals=texturefilename_normals,
                                 texturefilename_ambient_occlusion=texturefilename_ambient_occlusion,
                                 texturefilename_specular=texturefilename_specular)
        assert material.specularpower == specularpower
        assert material.texture_scales == texture_scales
        assert material.texturefilename == texturefilename
        assert material.texturefilename_normals == texturefilename_normals
        assert material.texturefilename_ambient_occlusion == texturefilename_ambient_occlusion
        assert material.texturefilename_specular == texturefilename_specular