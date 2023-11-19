import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import cv2
from func_timer import timer

# Load a sample image
IMG = cv2.imread("labs/lab2/tests/monke_512.jpg")
#IMG = IMG.astype(np.float32) / 255.0

# Rotation angle
#angle = 360

# Nearest-neighbor interpolation
@timer
def rotate_nearest_neighbor(img, angle):
    return rotate(img, angle, order=0, reshape=True)
#rotated_nearest = rotate(IMG, angle, order=0, reshape=True)

# Bilinear interpolation
@timer
def rotate_bilinear(img, angle):
    return rotate(img, angle, order=1, reshape=True)
#rotated_bilinear = rotate(IMG, angle, order=1, reshape=True)

# Cubic interpolation
@timer
def rotate_bicubic(img, angle):
    return rotate(img, angle, order=3, reshape=True)
#rotated_cubic = rotate(IMG, angle, order=3, reshape=True)


#cv2.imwrite("labs/lab2/tests/rotate_nn.jpg", rotated_nearest)
cv2.imwrite("labs/lab2/tests/rotate_bl_60.jpg", rotate_bilinear(IMG, 60))
cv2.imwrite("labs/lab2/tests/rotate_bl_360.jpg", rotate_bilinear(IMG, 360))
#cv2.imwrite("labs/lab2/tests/rotate_bc.jpg", rotated_cubic)