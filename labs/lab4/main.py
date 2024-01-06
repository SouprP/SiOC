import numpy as np
import cv2
from func_timer import timer


DIR = "labs/lab4/"
IMAGE = DIR + "cactus.jpg"
LAMBDA = 4

def calculate_mse_and_mae(original: np.ndarray, restored: np.ndarray) -> tuple[np.float64, np.float64]:
    mse = np.mean(np.square(np.subtract(original.astype(np.float64), restored.astype(np.float64))))
    mae = np.mean(np.abs(original.astype(np.float64) - restored.astype(np.float64)))

    return np.round(np.float64(mse), 2), np.round(np.float64(mae), 2)

@timer
def nearest(image: np.ndarray, size: int) -> np.ndarray:
    interpolated = np.zeros((size, size, 3))
    img_size = image.shape[0:2][0]
    scale = size / img_size

    for y in range(size):
        for x in range(size):
            x_nearest, y_nearest = np.int16(np.round(x / scale)), np.int16(np.round(y / scale))

            if x_nearest == img_size:  # sometimes it has problem with indexes
                x_nearest -= 1

            if y_nearest == img_size:
                y_nearest -= 1

            interpolated[x][y] = image[x_nearest][y_nearest]

    return interpolated

@timer
def bilinear(image: np.ndarray, size: int) -> np.ndarray:
    interpolated = np.zeros((size, size, 3))
    img_size = image.shape[0:2][0]
    scale = size / img_size

    for y in range(size):
        for x in range(size):
            x_old = x / scale
            y_old = y / scale

            x1, y1 = min(np.int16(np.floor(x_old)), img_size - 1), min(np.int16(np.floor(y_old)), img_size - 1)
            x2, y2 = min(np.int16(np.ceil(x_old)), img_size - 1), min(np.int16(np.ceil(y_old)), img_size - 1)

            q11, q12 = image[x1][y1], image[x2][y1]
            q21, q22 = image[x1][y2], image[x2][y2]

            p1 = q12 * (x_old - np.floor(x_old)) + q11 * (1.0 - (x_old - np.floor(x_old)))
            p2 = q22 * (x_old - np.floor(x_old)) + q21 * (1.0 - (x_old - np.floor(x_old)))

            p = (1.0 - (y_old - np.floor(y_old))) * p2 + (y_old - np.floor(y_old)) * p1

            interpolated[x][y] = np.round(p)

    return interpolated


def u(s: float, a: float) -> float:
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a + 2) * (abs(s)**3) - (a + 3) * (abs(s)**2) + 1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a * (abs(s)**3) - (5 * a) * (abs(s)**2) + (8 * a) * abs(s)- 4 * a
    return 0


def padding(image: np.ndarray, height: int, width: int, n_colours: int = 3) -> np.ndarray:
    padded = np.zeros((height + 4, width + 4, n_colours))
    padded[2:height + 2, 2:width + 2, :n_colours] = image

    padded[2:height + 2, 0:2, :n_colours] = image[:, 0:1, :n_colours]
    padded[height + 2:height + 4, 2:width + 2, :] = image[height - 1:height, :, :]
    padded[2:height + 2, width + 2:width + 4, :] = image[:, width - 1:width, :]
    padded[0:2, 2:width + 2, :n_colours] = image[0:1, :, :n_colours]

    padded[0:2, 0:2, :n_colours] = image[0, 0, :n_colours]
    padded[height + 2:height + 4, 0:2, :n_colours] = image[height - 1, 0, :n_colours]
    padded[height + 2:height + 4, width + 2:width + 4, :n_colours] = image[height - 1, width - 1, :n_colours]
    padded[0:2, width + 2:width + 4, :n_colours] = image[0, width - 1, :n_colours]

    return padded

@timer
def keys(image: np.ndarray, size: int) -> np.ndarray:
    # function was taken from:
    # https://github.com/rootpine/Bicubic-interpolation/blob/master/bicubic.py

    interpolated = np.zeros((size, size, 3))
    img_size = image.shape[0:2][0]
    image = padding(image, img_size, img_size, 3)

    a = -0.5
    h = 1 / (size / img_size)

    for c in range(3):
        for j in range(size):
            for i in range(size):
                x, y = i * h + 2, j * h + 2

                x1 = 1 + x - np.floor(x)
                x2 = x - np.floor(x)
                x3 = np.floor(x) + 1 - x
                x4 = np.floor(x) + 2 - x

                y1 = 1 + y - np.floor(y)
                y2 = y - np.floor(y)
                y3 = np.floor(y) + 1 - y
                y4 = np.floor(y) + 2 - y

                mat_l = np.matrix([[u(x1, a), u(x2, a), u(x3, a), u(x4, a)]])
                mat_m = np.matrix([[image[np.int16(y - y1), np.int16(x - x1), c],
                                    image[np.int16(y - y2), np.int16(x - x1), c],
                                    image[np.int16(y + y3), np.int16(x - x1), c],
                                    image[np.int16(y + y4), np.int16(x - x1), c]],
                                   [image[np.int16(y - y1), np.int16(x - x2), c],
                                    image[np.int16(y - y2), np.int16(x - x2), c],
                                    image[np.int16(y + y3), np.int16(x - x2), c],
                                    image[np.int16(y + y4), np.int16(x - x2), c]],
                                   [image[np.int16(y - y1), np.int16(x + x3), c],
                                    image[np.int16(y - y2), np.int16(x + x3), c],
                                    image[np.int16(y + y3), np.int16(x + x3), c],
                                    image[np.int16(y + y4), np.int16(x + x3), c]],
                                   [image[np.int16(y - y1), np.int16(x + x4), c],
                                    image[np.int16(y - y2), np.int16(x + x4), c],
                                    image[np.int16(y + y3), np.int16(x + x4), c],
                                    image[np.int16(y + y4), np.int16(x + x4), c]]])

                mat_r = np.matrix([[u(y1, a)], [u(y2, a)], [u(y3, a)], [u(y4, a)]])

                interpolated[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)

    return interpolated

poisson_image = np.random.poisson(cv2.imread(IMAGE) / LAMBDA)

poisson_image_nearest = poisson_image
poisson_image_bilinear = poisson_image
poisson_image_keys = poisson_image

################################
poisson_image_nearest = nearest(poisson_image_nearest, 100)
poisson_image_nearest = nearest(poisson_image_nearest, 1024)
#poisson_image_nearest *= LAMBDA
ERROR = calculate_mse_and_mae(cv2.imread(IMAGE), poisson_image_nearest)
print("Kwadratowy: ", ERROR[0], ", Absolutny: ", ERROR[1])


poisson_image_bilinear = bilinear(poisson_image_bilinear, 100)
poisson_image_bilinear = bilinear(poisson_image_bilinear, 1024)
#poisson_image_bilinear = poisson_image_bilinear * LAMBDA
ERROR = calculate_mse_and_mae(cv2.imread(IMAGE), poisson_image_bilinear)
print("Kwadratowy: ", ERROR[0], ", Absolutny: ", ERROR[1])


poisson_image_keys = keys(poisson_image_keys, 100)
poisson_image_keys = keys(poisson_image_keys, 1024)
#poisson_image_keys *= LAMBDA
ERROR = calculate_mse_and_mae(cv2.imread(IMAGE), poisson_image_keys)
print("Kwadratowy: ", ERROR[0], ", Absolutny: ", ERROR[1])

# poisson_image = poisson_image * LAMBDA

#ERROR = calculate_mse_and_mae(cv2.imread(IMAGE), poisson_image)
#print("Kwadratowy: ", ERROR[0], ", Absolutny: ", ERROR[1])

cv2.imwrite(DIR + "Poisson_nearest_dark_4x.jpg", poisson_image_nearest)
cv2.imwrite(DIR + "Poisson_bilinear_dark_4x.jpg", poisson_image_bilinear)
cv2.imwrite(DIR + "Poisson_keys_dark_4x.jpg", poisson_image_keys)
