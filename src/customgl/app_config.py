"""Configuration helpers for the shadow mapping demo."""

import json
from pathlib import Path


class ShadowMappingConfig:
    """Load and expose the shadow mapping configuration from JSON files."""

    def __init__(self, json_path: Path):
        with open(json_path, encoding="utf-8") as handle:
            data = json.load(handle)

        self.scene_name: str = data["Scene"]

        lights_path = json_path.parent / data["light_configuration"]
        with open(lights_path, encoding="utf-8") as handle:
            self.lights_data: list = json.load(handle)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scene_name={self.scene_name!r})"
