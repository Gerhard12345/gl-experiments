from customgl import Scene
import numpy as np
from customgl import MeshedSurfaceWithNormalOffset, MeshedSurfaceWall
from customgl import RoomDefinition, build_room

from customgl import (
    WoodenCeiling,
    GoldFoil,
    MuddyConcrete,
    TerraCottaTiles,
    WhiteBricks,
    WornMetal,
)

from .rollingsphereonsurface import RollingSphereOnSurface
from surfaces import AnalyticalDomain, ParametricSurface

class Scene4(Scene):
    dangle = 0.01
    rotz = 45
    roty = 35.26

    def __init__(self):
        super(Scene4, self).__init__()

        def surface_f(rphi):
            return 2 + 0.125 * (rphi[0] - 8) ** 2 - 0.5 * np.sin(2 * rphi[1])

        def surface_df(rphi):
            return [2 * 0.125 * (rphi[0] - 8), -2 * 0.5 * np.cos(2 * rphi[1])]

        bounds = [[6, 10], [0, 2 * np.pi]]
        x0 = 9
        y0 = 2 * np.pi - 0.4
        circular_analytical_domain = AnalyticalDomain(
            lambda u, v: u * np.cos(v),
            lambda u, v: u * np.sin(v),
            lambda u, v: np.matrix([[np.cos(v), np.sin(v)], [-u * np.sin(v), u * np.cos(v)]]),
            *bounds,
        )
        z = surface_f([x0, y0])
        r = 0.5
        p = ParametricSurface(circular_analytical_domain, surface_f, surface_df)
        s3 = MeshedSurfaceWithNormalOffset(
            surface=p,
            h_u=0.0125,
            h_v=0.0125,
            position=np.array([0, 0, 0]),
            material=TerraCottaTiles(texture_scales=[0.1, 0.1]),
            offset=r,
        )
        self.objects.append(s3)
        s4 = MeshedSurfaceWall(s3, material=WhiteBricks(texture_scales=[0.5, 0.2]), bottom_height=-3)
        self.objects.append(s4)
        sphere_3d_position = [x0, y0, z]
        r = RollingSphereOnSurface(
            p,
            position=np.array(sphere_3d_position),
            material=GoldFoil(texture_scales=[3, 3]),
            r=r,
        )
        self.objects.append(r)
        room_definition = RoomDefinition(
            x=18,
            y=6.5,
            z=18,
            bottom_material=MuddyConcrete(texture_scales=[1, 1]),
            top_material=WoodenCeiling(texture_scales=[1, 1]),
            left_material=WornMetal(texture_scales=[4, 4 * 6.5 / 18]),
            right_material=WornMetal(texture_scales=[4, 4 * 6.5 / 18]),
            front_material=WornMetal(texture_scales=[4, 4 * 6.5 / 18]),
            back_material=WornMetal(texture_scales=[4, 4 * 6.5 / 18]),
            position=[0, 3.5, 0],
        )
        object_views = build_room(room_definition)
        self.objects.extend(object_views)
