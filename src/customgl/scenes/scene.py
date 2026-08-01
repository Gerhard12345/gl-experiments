from dataclasses import dataclass
from typing import List

import numpy as np
from customgl import Quad, Object3d
from customgl import Material


@dataclass
class RoomDefinition:
    x: float = None
    y: float = None
    z: float = None
    left_material: Material = None
    right_material: Material = None
    front_material: Material = None
    back_material: Material = None
    bottom_material: Material = None
    top_material: Material = None
    position: List[int] = None


class Scene:
    def __init__(self):
        self.n_lights = 4
        self.objects: List[Object3d] = []

    def update(self):
        for current_object in self.objects:
            current_object.update()


def build_room(roomdefinition: RoomDefinition) -> List[Object3d]:
    objects = []
    center_position = roomdefinition.position
    for x_position, angle, material in zip(
        [-roomdefinition.x, roomdefinition.x], [np.radians(-90), np.radians(90)], [roomdefinition.left_material, roomdefinition.right_material]
    ):
        q = Quad(position=np.array([x_position, 0, 0]), material=material, scale=np.array([1, 1, 1]))
        q.local_rot_y(angle)
        q.scale([1, roomdefinition.y, roomdefinition.z])
        objects.append(q)

    for y_position, angle, material in zip(
        [-roomdefinition.y, roomdefinition.y], [np.radians(90), np.radians(-90)], [roomdefinition.bottom_material, roomdefinition.top_material]
    ):
        q = Quad(position=np.array([0, y_position, 0]), material=material, scale=np.array([1, 1, 1]))
        q.local_rot_x(angle)
        q.scale([roomdefinition.x, 1, roomdefinition.z])
        objects.append(q)

    for z_position, angle, material in zip(
        [roomdefinition.z, -roomdefinition.z], [np.radians(0), np.radians(180)], [roomdefinition.front_material, roomdefinition.back_material]
    ):
        q = Quad(position=np.array([0, 0, z_position]), material=material, scale=np.array([1, 1, 1]))
        q.local_rot_y(angle)
        q.scale([roomdefinition.x, roomdefinition.y, 1])
        objects.append(q)
    for myobject in objects:
        myobject.translate(center_position)
    return objects


