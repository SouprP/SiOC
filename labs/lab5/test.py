import numpy as np
import cv2
import matplotlib.pyplot as plt
from func_timer import timer


def box(size: int) -> np.ndarray:
    return np.ones((size, size)) * (1 / size ** 2)


def gaussian(size: int, sigma: float = 1) -> np.ndarray:
    center = size // 2
    kernel = np.array([[np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
                        for x in range(size)] for y in range(size)])

    return kernel / np.sum(kernel)


def median(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[0:2]
    radius = size // 2
    padded = np.pad(image, pad_width=((radius, radius), (radius, radius), (0, 0)), mode='edge')
    denoised = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            for c in range(3):
                denoised[y, x, c] = np.median(padded[y:y + size, x:x + size, c].flatten())

    return denoised


def convolution2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    height, width = image.shape[0:2]
    ks = kernel.shape[0] // 2
    padded = np.pad(image, pad_width=((ks, ks), (ks, ks), (0, 0)), mode='edge')
    convolved = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            for c in range(3):
                convolved[y, x, c] = np.sum(padded[y:y + kernel.shape[0], x:x + kernel.shape[0], c] * kernel)

    return convolved


KERNELS = {
    'box': box,
    'gaussian': gaussian
}


# size - większy od 3 oraz nieparzysty
@timer
def denoise(image: np.ndarray, size: int, f: str = "") -> np.ndarray:
    return convolution2d(image, KERNELS[f](size)) if f else median(image, size)

DIR = "labs/lab5/"
POISON_1X = DIR + "Poisson_keys.jpg"
POISON_4X =  DIR +"Poisson_keys_4x.jpg"
POISON_16X = DIR + "Poisson_keys_16x.jpg"

IMAGE = DIR + "cactus.jpg"
#IMAGE = DIR + "Poisson_keys_16x.jpg"
LAMBDA = 4

def calculate_mse_and_mae(original: np.ndarray, restored: np.ndarray) -> tuple[np.float64, np.float64]:
    mse = np.mean(np.square(np.subtract(original.astype(np.float64), restored.astype(np.float64))))
    mae = np.mean(np.abs(original.astype(np.float64) - restored.astype(np.float64)))

    return np.round(np.float64(mse), 2), np.round(np.float64(mae), 2)

#arr1 = calculate_mse_and_mae(CACTUS, cactus_box)
#arr2 = calculate_mse_and_mae(CACTUS, cactus_gauss)
#print("Box, kwadratowy: ", arr1[0], "       , absolutny: ", arr1[1])
#print("Gaussian, kwadratowy: ", arr2[0], "       , absolutny: ", arr2[1])
#cv2.imwrite(DIR + "splot_box.jpg", cactus_box)
#cv2.imwrite(DIR + "splot_gaussian.jpg", cactus_gauss)

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

#########################
# WSPÓŁCZYNNIK K i P
k = 1
P = 8
#########################

K = 2**k * 3
print("K: ", K)

# WSPÓŁCZYNNIK P
# P = 7 => 128
# P = 8 => 256
#CACTUS = cv2.imread(IMAGE)
ORG = cv2.imread(IMAGE)
ORG = cv2.cvtColor(ORG, cv2.COLOR_BGR2RGB)
CACTUS = cv2.imread(POISON_16X)
CACTUS = cv2.cvtColor(CACTUS, cv2.COLOR_BGR2RGB)

#cactus_box = denoise(CACTUS, K, "box")
cactus_gauss = denoise(CACTUS, K, "gaussian")

#cactus_box_nearest = cactus_box
cactus_gauss_nearest = cactus_gauss
cactus_gauss_bilinear = cactus_gauss
cactus_gauss_keys = cactus_gauss

################################
#cactus_box_nearest = nearest(cactus_box_nearest, 2**8)
#cactus_box_nearest = nearest(cactus_box_nearest, 1024)
#ERROR = calculate_mse_and_mae(CACTUS, cactus_box_nearest)
#print("BOX nearest, Kwadratowy: ", ERROR[0], ", Absolutny: ", ERROR[1])


################################

#cactus_gauss_nearest_small = nearest(cactus_gauss_nearest, 2**P)
#cactus_gauss_nearest_normal = nearest(cactus_gauss_nearest_small, 1024)
#ERROR = calculate_mse_and_mae(CACTUS, cactus_gauss_nearest_normal)
#print("Gauss nearest, Kwadratowy: ", ERROR[0], ", Absolutny: ", ERROR[1])
#FINAL_IMAGE = cactus_gauss_nearest_normal

#cactus_gauss_bilinear_small = bilinear(cactus_gauss_bilinear, 2**P)
#cactus_gauss_bilinear_normal = bilinear(cactus_gauss_bilinear_small, 1024)
#ERROR = calculate_mse_and_mae(CACTUS, cactus_gauss_bilinear_normal)
#print("Gauss biinear, Kwadratowy: ", ERROR[0], ", Absolutny: ", ERROR[1])
#FINAL_IMAGE = cactus_gauss_bilinear_normal

cactus_gauss_keys_small = keys(cactus_gauss_keys, 2**P)
cactus_gauss_keys_normal = keys(cactus_gauss_keys_small, 1024)
ERROR = calculate_mse_and_mae(CACTUS, cactus_gauss_keys_normal)
print("Gauss Keys'a, Kwadratowy: ", ERROR[0], ", Absolutny: ", ERROR[1])
FINAL_IMAGE = cactus_gauss_keys_normal

fig, axs = plt.subplots(1, 3)
axs[0].imshow(CACTUS)
axs[0].set_title('ORYGINALNY')
axs[0].set_xticks([])
axs[0].set_yticks([])

axs[1].imshow(cactus_gauss)
axs[1].set_title('FILTR')
axs[1].set_xticks([])
axs[1].set_yticks([])

axs[2].imshow(CACTUS - FINAL_IMAGE)
axs[2].set_title('RÓŻNICA')
axs[2].set_xticks([])
axs[2].set_yticks([])

#axs[3].imshow(cactus_gauss - cactus_gauss_nearest_normal)
#axs[3].set_title('POWIĘKSZONY')
#axs[3].set_xticks([])
#axs[3].set_yticks([])
plt.subplots_adjust(0.025, 0, 0.975, 1, hspace=0.01, wspace=0.05)

#plt.savefig(DIR + 'nn_k' + str(k) + '_p' + str(P))
#plt.savefig(DIR + 'lin_k' + str(k) + '_p' + str(P))
#plt.savefig(DIR + 'keys_k' + str(k) + '_p' + str(P))
plt.savefig(DIR + 'keys_poisson16x_k' + str(k) + '_p' + str(P))

plt.show()