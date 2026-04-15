import json
from pathlib import Path


class ShadowMappingConfig:
    """
    Loads shadow_mapping.json and the nested light configuration it references.

    shadow_mapping.json shape::

        {
            "Scene": "Scene4",
            "light_configuration": "guielements/scene4_lights.json"
        }

    The ``light_configuration`` path is resolved relative to the config file.
    """

    def __init__(self, json_path: Path):
        with open(json_path) as f:
            data = json.load(f)

        self.scene_name: str = data["Scene"]

        lights_path = json_path.parent / data["light_configuration"]
        with open(lights_path) as f:
            self.lights_data: list = json.load(f)
