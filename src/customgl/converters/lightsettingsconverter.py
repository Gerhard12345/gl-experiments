from typing import Dict, List

from ..drawing.lights import Lights
from ..guielements.tabview import LightDef, SliderGroupDef


class LightSettingsConverter:
    """
    Converts a List[LightDef] (from LightingControlPanel._TAB_DEFS) into a
    populated Lights instance for use in scenes.

    LightDef entries with type "Directional" feed set_directional_lights();
    entries with type "Point" feed set_point_lights().
    Color slider values [0, 255] are normalised to [0.0, 1.0].

    Usage::

        lights = LightSettingsConverter(panel._TAB_DEFS).to_lights()
    """

    def __init__(self, tab_defs: List[LightDef]):
        self._tab_defs = tab_defs

    def to_lights(self) -> Lights:
        lights = Lights()

        ambient_lights = [l for l in self._tab_defs if l.type == "Ambient"]
        dir_lights = [l for l in self._tab_defs if l.type == "Directional"]
        pt_lights  = [l for l in self._tab_defs if l.type == "Point"]

        if ambient_lights:
            color = self._normalize_rgb(self._group_values(ambient_lights[0].light_properties["Color"]))
            lights.set_ambient_light(color)

        if dir_lights:
            lights.set_directional_lights(
                positions=[list(self._group_values(l.light_properties["Direction"]).values()) for l in dir_lights],
                diffuse  =[self._normalize_rgb(self._group_values(l.light_properties["Diffuse"])) for l in dir_lights],
                specular =[self._normalize_rgb(self._group_values(l.light_properties["Specular"])) for l in dir_lights],
            )

        if pt_lights:
            n = len(pt_lights)
            lights.set_point_lights(
                positions=[list(self._group_values(l.light_properties["Position"]).values()) for l in pt_lights],
                diffuse  =[self._normalize_rgb(self._group_values(l.light_properties["Diffuse"])) for l in pt_lights],
                specular =[self._normalize_rgb(self._group_values(l.light_properties["Specular"])) for l in pt_lights],
                constant =[1.0]  * n,
                linear   =[0.09] * n,
                quadratic=[0.032]* n,
            )

        return lights

    @staticmethod
    def _group_values(group_def: SliderGroupDef) -> Dict[str, int]:
        return dict(zip(group_def.slider_config.keys(), group_def.default_values))

    @staticmethod
    def _normalize_rgb(channel: Dict[str, int]) -> List[float]:
        return [channel["R"] / 255.0, channel["G"] / 255.0, channel["B"] / 255.0]
