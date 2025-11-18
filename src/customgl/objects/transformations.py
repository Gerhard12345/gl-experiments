import numpy as np
from numpy.typing import NDArray


class Transformations:
    @staticmethod
    def scalemat(scale_xyz: NDArray[np.float32]):
        return np.matrix([[scale_xyz[0], 0, 0, 0], [0, scale_xyz[1], 0, 0], [0, 0, scale_xyz[2], 0], [0, 0, 0, 1]]).transpose()

    @staticmethod
    def translationmat(position):
        return np.matrix([[1, 0, 0, position[0]], [0, 1, 0, position[1]], [0, 0, 1, position[2]], [0, 0, 0, 1]]).transpose()

    @staticmethod
    def localrotationmat_axis(position, angle, axis):
        localrotmat = (
            Transformations.translationmat(-position) * Transformations.rotationmat_axis(angle, axis) * Transformations.translationmat(position)
        )
        return localrotmat

    @staticmethod
    def rotationmat_axis(angle, axis):
        axis /= np.linalg.norm(axis)
        u_x, u_y, u_z = axis
        rotmat_axis = np.matrix(
            [
                [
                    u_x**2 * (1 - np.cos(angle)) + np.cos(angle),
                    u_x * u_y * (1 - np.cos(angle)) - u_z * np.sin(angle),
                    u_x * u_z * (1 - np.cos(angle)) + u_y * np.sin(angle),
                    0,
                ],
                [
                    u_x * u_y * (1 - np.cos(angle)) + u_z * np.sin(angle),
                    u_y**2 * (1 - np.cos(angle)) + np.cos(angle),
                    u_z * u_y * (1 - np.cos(angle)) - u_x * np.sin(angle),
                    0,
                ],
                [
                    u_x * u_z * (1 - np.cos(angle)) - u_y * np.sin(angle),
                    u_z * u_y * (1 - np.cos(angle)) + u_x * np.sin(angle),
                    u_z**2 * (1 - np.cos(angle)) + np.cos(angle),
                    0,
                ],
                [0, 0, 0, 1],
            ]
        ).transpose()
        return rotmat_axis


def getCentralProjectionMatrix(asize, znear=0.1, zfar=100, fov=np.tan(np.pi / 8)):
    aspect = 1
    if asize[1] != 0:
        aspect = asize[0] / asize[1]
    projectionmat = np.matrix(
        [
            [1 / (aspect * fov), 0.0, 0.0, 0.0],
            [0, 1 / fov, 0.0, 0.0],
            [0, 0, (zfar + znear) / (znear - zfar), 2 * zfar * znear / (znear - zfar)],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    ).transpose()
    return projectionmat


# def getOrthogonalProjectionMatrix(asize, bottom=-18, top=8, znear=5, zfar=35):
def getOrthogonalProjectionMatrix(asize, bottom=-28, top=28, znear=-5, zfar=45):
    aspect = 1
    if asize[1] != 0:
        aspect = asize[0] / asize[1]
    total_height = top - bottom
    left = bottom * aspect
    right = top * aspect
    left = -0.5 * total_height * aspect
    right = 0.5 * total_height * aspect
    projectionmat = np.matrix(
        [
            [2 / (right - left), 0.0, 0.0, -(right + left) / (right - left)],
            [0, 2 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)],
            [0, 0, -2 / (zfar - znear), -(zfar + znear) / (zfar - znear)],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    ).transpose()
    return projectionmat
