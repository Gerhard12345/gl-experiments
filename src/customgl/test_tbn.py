import numpy as np

normals = [[0, 0, -1], [0, 0, 1], [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]]
tangents = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]
bitangents = [[0, 1, 0], [0, 1, 0], [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1]]
for tangent, bitangent, normal in zip(tangents, bitangents, normals):
    tbn = np.array([tangent, bitangent, normal]).T
    print(tangent, bitangent, normal, "\n", tbn, np.linalg.det(tbn))
    print()

print()

for tangent, bitangent, normal in zip(tangents, bitangents, normals):
    tbn = np.array([tangent, bitangent, normal]).T
    n = tbn @ np.array([[0, 0, 1]]).T
    print(n, normal)
    print()


from imageio import imread, imwrite
import scipy as sp
from scipy import ndimage


def smooth_gaussian(im: np.ndarray, sigma) -> np.ndarray:

    if sigma == 0:
        return im

    im_smooth = im.astype(float)
    kernel_x = np.arange(-3 * sigma, 3 * sigma + 1).astype(float)
    kernel_x = np.exp((-(kernel_x**2)) / (2 * (sigma**2)))

    im_smooth = ndimage.convolve(im_smooth, kernel_x[np.newaxis])

    im_smooth = ndimage.convolve(im_smooth, kernel_x[np.newaxis].T)

    return im_smooth


def sobel(im_smooth):
    gradient_x = im_smooth.astype(float)
    gradient_y = im_smooth.astype(float)

    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    gradient_x = ndimage.convolve(gradient_x, kernel)
    gradient_y = ndimage.convolve(gradient_y, kernel.T)
    return gradient_x, gradient_y


im = imread("customgl/textures/brickwall.jpg")
# im1 = np.sum(im, axis=2) / 3
im1 = im[..., 0] * 0.3 + im[..., 1] * 0.6 + im[..., 2] * 0.1
im1 = smooth_gaussian(im1, sigma=2)
dx, dy = sobel(im1)
dx, dy = sobel(dx + dy)
# dy = im1[2:, 1:-1] - im1[0:-2, 1:-1]
# dx = im1[1:-1, 2:] - im1[1:-1, 0:-2]
mx = np.max(np.abs(dx))
my = np.max(np.abs(dy))
m = np.max([mx, my])
dx /= m
dy /= m
dz = np.ones(dx.shape) * 0.25

norm = np.sqrt(dx**2 + dy**2 + dz**2)
a = np.zeros((*dx.shape, 3))
n_x = dx / norm
n_y = -dy / norm
n_z = dz / (norm)
a[:, :, 0] = 0.5 * (n_x + 1)
a[:, :, 1] = 0.5 * (n_y + 1)
a[:, :, 2] = n_z

import matplotlib.pyplot as plt

# plt.imshow(a)

# imwrite("temp.jpg", (255 * a).astype(np.uint8))


# im = imread("customgl/textures/wood1.jpg")
# im_normal = imread("customgl/textures/wood1_normal.jpg")
# im = im[:1024, :1024, :]
# im_normal = im_normal[:1024, :1024, :]
# imwrite("customgl/textures/wood1.jpg", im)
# imwrite("customgl/textures/wood1_normal.jpg", im_normal)
