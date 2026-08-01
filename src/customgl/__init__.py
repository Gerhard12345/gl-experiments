import sys

from .objects import material as _material_module
from .objects.material import *
from .objects.surface import *
from .objects.transformations import Transformations
from .objects.objects3d import Cube, Quad, Object3d, SphericalCoordianteSphere
from .scenes.scene import Scene, RoomDefinition, build_room

import customgl.objects.surface as surface