import numpy as np
from typing import List
from .transformations import getCentralProjectionMatrix


class Camera:
    def __init__(self, eye=np.array([0, 0, 2]), at=np.array([0, 0, 0]), up=[0, 1, 0], fov: float = np.pi * 0.5, near=0.1, far=100):
        self.fov = fov
        self.at = np.array(at, dtype=np.float64)
        self.eye = np.array(eye, dtype=np.float64)
        self.up = np.array(up, dtype=np.float64)
        at = np.array(at)
        self.d = np.linalg.norm(eye - at)
        direction = 1 / self.d * (eye - at)
        self.phi = np.arctan2(direction[2], direction[0])
        self.theta = np.arccos(direction[1])
        self.near = near
        self.far = far
        self.lookAt()

    def lookAt(self):
        at = self.at
        eye = self.eye
        up = self.up
        zaxis = (at - eye) / np.linalg.norm(at - eye)
        xaxis = np.cross(zaxis, up) / np.linalg.norm(np.cross(zaxis, up))
        yaxis = np.cross(xaxis, zaxis)

        zaxis *= -1
        self.modelmat = np.matrix(
            [
                np.append(xaxis, -np.dot(xaxis, eye)),
                np.append(yaxis, -np.dot(yaxis, eye)),
                np.append(zaxis, -np.dot(zaxis, eye)),
                np.array([0, 0, 0, 1]),
            ],
            dtype=np.float32,
        ).transpose()

    def getViewmat(self):
        return self.modelmat

    def getViewingPosition(self):
        return self.eye

    def getProjectionmat(self, viewing_width: int, viewing_height: int):
        return getCentralProjectionMatrix((viewing_width, viewing_height), znear=self.near, zfar=self.far, fov=self.fov)

    def rotate_theta(self, dtheta):
        self.theta += np.deg2rad(dtheta)
        self.eye = self.at + self.d * np.array(
            [np.cos(self.phi) * np.sin(self.theta), np.cos(self.theta), np.sin(self.phi) * np.sin(self.theta)]
        )
        self.lookAt()

    def rotate_phi(self, dphi):
        self.phi += np.deg2rad(dphi)
        self.eye = self.at + self.d * np.array(
            [
                np.cos(self.phi) * np.sin(self.theta),
                np.cos(self.theta),
                np.sin(self.phi) * np.sin(self.theta),
            ]
        )
        self.lookAt()

    def zoom(self, zoom_factor: float):
        direction = (self.at - self.eye) * zoom_factor
        self.eye = self.at - direction
        self.d = np.linalg.norm(self.eye - self.at)
        self.lookAt()

    def translate(self, translation: List[float]):
        self.eye += np.array([translation[0], 0.0, translation[1]])
        self.at += np.array([translation[0], 0.0, translation[1]])
        self.lookAt()

    def set_lookat_position(self, position: List[float]):
        self.at = position[0:3]
        d = self.eye - self.at
        self.d = np.linalg.norm(d)
        d /= self.d
        self.phi = np.atan2(d[2], d[0])
        self.theta = np.acos(d[1])

    def update(self):
        pass


class Camera1(Camera):
    def __init__(self, eye=np.array([0, 0, 2]), at=np.array([0, 0, 0]), up=[0, 1, 0]):
        super().__init__(eye=eye, at=at, up=up)
        fac = 1
        self.fov = np.tan(np.pi / 12) / fac
        self.i = 0

    def update(self):
        self.i += 1
        omega = 0.005 * self.i
        fac = 1.5 + 0.5 * np.sin(omega)
        self.eye = np.array([0, 3 * fac, 18 * fac], dtype=np.float64)
        self.at = np.array([0, 0, 0], dtype=np.float64)
        self.up = np.array([0, 1, 0], dtype=np.float64)
        # set field of view such that ||eye||^2/fov = const
        self.fov = np.tan(np.pi / 12) / fac
