from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..objects.camera import Camera

_CUBE_FACE_DIRECTIONS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
_CUBE_FACE_UP_VECTORS_DEFAULT = [[0, -1, 0], [0, -1, 0], [0, 0, -1], [0, 0, -1], [0, -1, 0], [0, -1, 0]]


@dataclass
class DirectionalLight:
    light_space_camera: Camera = None
    ambient: List[float] = None
    diffuse: List[float] = None
    specular: List[float] = None

    def __post_init__(self):
        self.light_space_camera.lookAt()

    @property
    def direction(self):
        return -self.light_space_camera.getViewingPosition()


@dataclass
class PointLight:
    light_space_camera: List[Camera] = None
    ambient: List[float] = None
    diffuse: List[float] = None
    specular: List[float] = None
    constant: float = None
    linear: float = None
    quadratic: float = None

    def __post_init__(self):
        for camera in self.light_space_camera:
            camera.lookAt()

    @property
    def position(self):
        return self.light_space_camera[0].getViewingPosition()


class Lights:
    def __init__(self):
        self._lights: List[DirectionalLight] = []
        self._point_lights: List[PointLight] = []

    @property
    def lights(self) -> List[DirectionalLight]:
        return self._lights

    @property
    def point_lights(self) -> List[PointLight]:
        return self._point_lights

    def set_lights(self, lights: List[DirectionalLight], point_lights: List[PointLight]) -> None:
        self._lights = lights
        self._point_lights = point_lights

    def set_directional_lights(
        self,
        positions: List,
        ambient: List[List[float]],
        diffuse: List[List[float]],
        specular: List[List[float]],
    ) -> None:
        self._lights = [
            DirectionalLight(light_space_camera=Camera(eye=pos), ambient=amb, diffuse=diff, specular=spec)
            for pos, amb, diff, spec in zip(positions, ambient, diffuse, specular)
        ]

    def set_point_lights(
        self,
        positions: List,
        ambient: List[List[float]],
        diffuse: List[List[float]],
        specular: List[List[float]],
        constant: List[float],
        linear: List[float],
        quadratic: List[float],
        near: float = 1,
        far: float = 25,
        up_vectors: Optional[List[List[float]]] = None,
    ) -> None:
        if up_vectors is None:
            up_vectors = _CUBE_FACE_UP_VECTORS_DEFAULT
        self._point_lights = [
            PointLight(
                light_space_camera=[
                    Camera(eye=np.array(pos), at=np.array(pos) + np.array(direction), up=up, fov=0.5 * np.pi, near=near, far=far)
                    for direction, up in zip(_CUBE_FACE_DIRECTIONS, up_vectors)
                ],
                ambient=amb,
                diffuse=diff,
                specular=spec,
                constant=const,
                linear=lin,
                quadratic=quad,
            )
            for pos, amb, diff, spec, const, lin, quad in zip(
                positions, ambient, diffuse, specular, constant, linear, quadratic
            )
        ]
