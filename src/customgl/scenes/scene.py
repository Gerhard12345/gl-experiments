from dataclasses import dataclass
from typing import List

import numpy as np

from ..objects.objects3d import Cube
from ..objects.objects3d import Quad
from ..objects.objects3d import Object3d
from ..objects.camera import Camera
from ..objects.material import (
    Material,
    WoodenCeiling,
    BrickWall1,
    BrickWall2,
    Marble1,
    GoldFoil,
    MuddyConcrete,
    TerraCottaTiles,
    Wood1,
    MetalPanel1,
    WhiteBricks,
    WornMetal,
)
from ..objects.transformations import Transformations
from ..objects.rolling_sphere import RollingSphere, RollingSphereParametric
from ..objects.surface import (
    MeshedSurfaceWithNormalOffset,
    MeshedSurfaceWall,
    AnalyticalDomain,
)


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


class Scene:
    def __init__(self):
        self.n_lights = 4
        self.objects: List[Object3d] = []
        self.lights: List[DirectionalLight] = []
        self.point_lights: List[PointLight] = []
        self.set_lights()

    def set_lights(self):
        pass

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


class Scene1(Scene):
    dangle = 0.01
    rotz = 45
    roty = 35.26

    def set_lights(self):
        unidirectional_lights_position = [[0, 3, 2], [0, 0.5, 2], [0, 3, 2], [0, 0.5, 5]]
        lights_ambient = [[0.5, 0.5, 0.5]] * 4
        lights_diffuse = [[0.8, 0.8, 0.8]] * 4
        lights_specular = [[1, 1, 1]] * 4
        lights_constant = [1] * 4
        lights_linear = [0.09] * 4
        lights_quadratic = [0.032] * 4
        self.lights = [
            DirectionalLight(
                light_space_camera=Camera(eye=light_position), ambient=light_ambient, diffuse=light_diffuse, specular=light_specular
            )
            for light_position, light_ambient, light_diffuse, light_specular, light_constant, light_linear, light_quadratic in zip(
                unidirectional_lights_position,
                lights_ambient,
                lights_diffuse,
                lights_specular,
                lights_constant,
                lights_linear,
                lights_quadratic,
            )
        ]

        point_lights_position = [np.array([0, 3, 2]), np.array([0, 0.5, 2]), np.array([0, 0.5, 2]), np.array([0, 3, 2])]
        lights_ambient = [[1.0, 1.0, 1.0]] * 4
        lights_diffuse = [[1.0, 1.0, 1.0]] * 4
        lights_specular = [[1, 1, 1]] * 4
        lights_constant = [1] * 4
        lights_linear = [0.09] * 4
        lights_quadratic = [0.032] * 4
        self.point_lights = [
            PointLight(
                light_space_camera=[
                    Camera(eye=light_position, at=light_position + direction, up=up_vec, fov=0.5 * np.pi, near=1, far=25)
                    for direction, up_vec in zip(
                        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
                        [[0, -1, 0], [0, -1, 0], [0, 0, -1], [0, 0, -1], [0, -1, 0], [0, -1, 0]],
                    )
                ],
                ambient=light_ambient,
                diffuse=light_diffuse,
                specular=light_specular,
                constant=light_constant,
                linear=light_linear,
                quadratic=light_quadratic,
            )
            for light_position, light_ambient, light_diffuse, light_specular, light_constant, light_linear, light_quadratic in zip(
                point_lights_position,
                lights_ambient,
                lights_diffuse,
                lights_specular,
                lights_constant,
                lights_linear,
                lights_quadratic,
            )
        ]

    def __init__(self):
        super().__init__()
        # model 0
        c = Cube(
            position=np.array([0, 0, -15]),
            scale=np.array([3, 3, 3]),
            material=BrickWall1(),
        )
        c.local_rot_z(np.radians(self.rotz)).local_rot_x(np.radians(self.roty))
        c.add_dynamic(Transformations.localrotationmat_axis(c.position, self.dangle, np.array([0, 1.0, 0])))
        self.objects.append(c)
        c = Cube(position=np.array([0, 0, -2]), scale=np.array([1, 1, 1]), material=Marble1())
        c.add_dynamic(Transformations.localrotationmat_axis(c.position, -5 * self.dangle, np.array([0, 1.0, 0])))
        self.objects.append(c)
        # model 1
        positions = np.array(
            [[-3, 2, -0.5], [3, 2, -0.5], [-3, -2, -0.5], [3, -2, -0.5], [-3, 2, -14.5], [3, 2, -14.5], [-3, -2, -14.5], [3, -2, -14.5]]
        )
        materials = [
            Wood1(),
            TerraCottaTiles(),
            GoldFoil(),
            MetalPanel1(),
        ] * 2
        for position, material in zip(positions, materials):

            c = Cube(position=position, scale=np.array([1, 1, 1]), material=material)
            c.local_rot_z(np.radians(self.rotz)).local_rot_x(np.radians(self.roty))
            c.add_dynamic(Transformations.localrotationmat_axis(c.position, self.dangle, np.array([0, 1.0, 0])))
            self.objects.append(c)

        room_definition = RoomDefinition(
            x=7.5,
            y=7.5,
            z=40,
            bottom_material=MuddyConcrete(texture_scales=[2, 2 * 40 / 7.5]),
            top_material=WoodenCeiling(texture_scales=[2, 2 * 40 / 7.5]),
            left_material=BrickWall2(texture_scales=[2 * 40 / 7.5, 2]),
            right_material=BrickWall2(texture_scales=[2 * 40 / 7.5, 2]),
            front_material=BrickWall2(texture_scales=[2, 2]),
            back_material=BrickWall2(texture_scales=[2, 2]),
            position=[0, 0, 0],
        )
        object_views = build_room(room_definition)
        self.objects.extend(object_views)


class Scene3(Scene):
    dangle = 0.01
    rotz = 45
    roty = 35.26

    def __init__(self):
        super().__init__()

        def surface_f(q):
            return (np.sin(q[0]) * (3 - 0.3 * q[0])) + 0.15 * q[1] ** 2

        def surface_df(q):
            return [-0.3 * np.sin(q[0]) + (3 - 0.3 * q[0]) * np.cos(q[0]), 2 * 0.15 * q[1]]

        bounds = [[1, 23], [-3, 3]]
        x0 = np.pi / 2
        y0 = 0.25
        b = AnalyticalDomain(lambda u, v: u, lambda u, v: v, lambda u, v: np.matrix([[1, 0], [0, 1]]), bounds[0], bounds[1])
        z = surface_f([x0, y0])
        r = 0.5
        s3 = MeshedSurfaceWithNormalOffset(
            analytical_domain=b,
            f_z=lambda u, v: surface_f([u, v]),
            d_f=lambda u, v: surface_df([u, v]),
            h_u=0.0125,
            h_v=0.025,
            position=np.array([0, 0, 0]),
            material=WoodenCeiling(texture_scales=[22 / 6, 1]),
            # material=WoodenCeiling(texture_scales=[22 / 6, 1]),
            r=r,
        )

        self.objects.append(s3)
        r = RollingSphere(
            surface_f=surface_f,
            surface_df=surface_df,
            position=np.array([x0, y0, z]),
            material=WoodenCeiling(texture_scales=[2, 2]),
            r=r,
        )
        self.objects.append(r)
        room_definition = RoomDefinition(
            x=6,
            y=7.5,
            z=11,
            bottom_material=MuddyConcrete(texture_scales=[2, 2 * 10 / 6]),
            top_material=WoodenCeiling(texture_scales=[2, 2 * 10 / 6]),
            left_material=WornMetal(texture_scales=[2 * 10 / 6, 2]),
            right_material=WornMetal(texture_scales=[2 * 10 / 6, 2]),
            front_material=WornMetal(texture_scales=[2, 2]),
            back_material=WornMetal(texture_scales=[2, 2]),
            position=[0, 3.5, 12],
        )
        object_views = build_room(room_definition)
        self.objects.extend(object_views)

    def set_lights(self):
        unidirectional_lights_position = [[0, 10, -14], [0, 10, -14], [0, 10, 14], [0, 10, 14]]
        lights_ambient = [[0.5, 0.5, 0.5]] * 4
        lights_diffuse = [[0.8, 0.8, 0.8]] * 4
        lights_specular = [[1, 1, 1]] * 4
        lights_constant = [1] * 4
        lights_linear = [0.09] * 4
        lights_quadratic = [0.032] * 4
        self.lights = [
            DirectionalLight(
                light_space_camera=Camera(eye=light_position),
                ambient=light_ambient,
                diffuse=light_diffuse,
                specular=light_specular,
                constant=light_constant,
                linear=light_linear,
                quadratic=light_quadratic,
            )
            for light_position, light_ambient, light_diffuse, light_specular, light_constant, light_linear, light_quadratic in zip(
                unidirectional_lights_position,
                lights_ambient,
                lights_diffuse,
                lights_specular,
                lights_constant,
                lights_linear,
                lights_quadratic,
            )
        ]
        point_lights_position = [np.array([6, 10, 12]), np.array([6, 10, 12]), np.array([-6, 10, 12]), np.array([-6, 10, 12])]
        lights_ambient = [[0.5, 0.5, 0.5]] * 4
        lights_diffuse = [[0.8, 0.8, 0.8]] * 4
        lights_specular = [[1, 1, 1]] * 4
        lights_constant = [1] * 4
        lights_linear = [0.029] * 4
        lights_quadratic = [0.0032] * 4
        self.point_lights = [
            PointLight(
                light_space_camera=[
                    Camera(eye=light_position, at=light_position + direction, up=up_vec, fov=0.5 * np.pi, near=1, far=25)
                    for direction, up_vec in zip(
                        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
                        [[0, -1, 0], [0, -1, 0], [0, 0, -1], [0, 0, -1], [0, -1, 0], [0, -1, 0]],
                    )
                ],
                ambient=light_ambient,
                diffuse=light_diffuse,
                specular=light_specular,
                constant=light_constant,
                linear=light_linear,
                quadratic=light_quadratic,
            )
            for light_position, light_ambient, light_diffuse, light_specular, light_constant, light_linear, light_quadratic in zip(
                point_lights_position,
                lights_ambient,
                lights_diffuse,
                lights_specular,
                lights_constant,
                lights_linear,
                lights_quadratic,
            )
        ]


class Scene4(Scene):
    dangle = 0.01
    rotz = 45
    roty = 35.26

    def __init__(self):
        super(Scene4, self).__init__()

        def surface_f(rphi):
            return 0.125 * (rphi[0] - 8) ** 2 - 0.5 * np.sin(2 * rphi[1])

        def surface_df(rphi):
            return [2 * 0.125 * (rphi[0] - 8), -2 * 0.5 * np.cos(2 * rphi[1])]

        bounds = [[6, 10], [0, 2 * np.pi]]
        x0 = 9
        y0 = 2 * np.pi - 0.4
        b = AnalyticalDomain(
            lambda u, v: u * np.cos(v),
            lambda u, v: u * np.sin(v),
            lambda u, v: np.matrix([[np.cos(v), np.sin(v)], [-u * np.sin(v), u * np.cos(v)]]),
            *bounds,
        )
        z = surface_f([x0, y0])
        r = 0.5
        s3 = MeshedSurfaceWithNormalOffset(
            analytical_domain=b,
            f_z=lambda u, v: surface_f([u, v]),
            d_f=lambda u, v: surface_df([u, v]),
            h_u=0.0125,
            h_v=0.0125,
            position=np.array([0, 0, 0]),
            material=TerraCottaTiles(texture_scales=[0.1, 0.1]),
            r=r,
        )
        self.objects.append(s3)
        s4 = MeshedSurfaceWall(s3, material=WhiteBricks(texture_scales=[2, 0.2]), bottom_height=-3)
        self.objects.append(s4)
        sphere_3d_position = [b.fx(x0, y0), b.fy(x0, y0), z]

        r = RollingSphereParametric(
            analytical_domain=b,
            surface_f=surface_f,
            surface_df=surface_df,
            position=np.array(sphere_3d_position),
            initial_parameter=[x0, y0],
            material=WoodenCeiling(texture_scales=[1, 1]),
            r=r,
        )
        self.objects.append(r)
        room_definition = RoomDefinition(
            x=12,
            y=5.5,
            z=12,
            bottom_material=MuddyConcrete(texture_scales=[2, 2 * 12 / 5.5]),
            top_material=WoodenCeiling(texture_scales=[2, 2 * 12 / 5.5]),
            left_material=WornMetal(texture_scales=[2 * 12 / 5.5, 2]),
            right_material=WornMetal(texture_scales=[2 * 12 / 5.5, 2]),
            front_material=WornMetal(texture_scales=[2, 2]),
            back_material=WornMetal(texture_scales=[2, 2]),
            position=[0, 2.5, 0],
        )
        object_views = build_room(room_definition)
        self.objects.extend(object_views)

    def set_lights(self):
        unidirectional_lights_position = [[-11, 16, -11], [-11, 16, 11], [11, 16, 11], [11, 16, -11]]
        lights_ambient = [[0.25, 0.25, 0.25]] * 4
        lights_diffuse = [[0.5, 0.5, 0.5]] * 4
        lights_specular = [[1, 1, 1]] * 4
        lights_constant = [1] * 4
        lights_linear = [0.09] * 4
        lights_quadratic = [0.032] * 4
        self.lights = [
            DirectionalLight(
                light_space_camera=Camera(eye=light_position), ambient=light_ambient, diffuse=light_diffuse, specular=light_specular
            )
            for light_position, light_ambient, light_diffuse, light_specular, light_constant, light_linear, light_quadratic in zip(
                unidirectional_lights_position,
                lights_ambient,
                lights_diffuse,
                lights_specular,
                lights_constant,
                lights_linear,
                lights_quadratic,
            )
        ]
        point_lights_position = [np.array([-11, 1, -11]), np.array([-11, 1, 11]), np.array([11, 1, -11]), np.array([11, 1, 11])]
        lights_ambient = [[0.75, 0.75, 0.75]] * 4
        lights_diffuse = [[1, 1, 1]] * 4
        lights_specular = [[1, 1, 1]] * 4
        lights_constant = [1] * 4
        lights_linear = [0.09] * 4
        lights_quadratic = [0.032] * 4
        self.point_lights = [
            PointLight(
                light_space_camera=[
                    Camera(eye=light_position, at=light_position + direction, up=up_vec, fov=0.5 * np.pi, near=1, far=50)
                    for direction, up_vec in zip(
                        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
                        [[0, -1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [0, -1, 0], [0, -1, 0]],
                    )
                ],
                ambient=light_ambient,
                diffuse=light_diffuse,
                specular=light_specular,
                constant=light_constant,
                linear=light_linear,
                quadratic=light_quadratic,
            )
            for light_position, light_ambient, light_diffuse, light_specular, light_constant, light_linear, light_quadratic in zip(
                point_lights_position,
                lights_ambient,
                lights_diffuse,
                lights_specular,
                lights_constant,
                lights_linear,
                lights_quadratic,
            )
        ]
