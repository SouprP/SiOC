import numpy as np
import cv2
from func_timer import timer
import scipy.ndimage

DIR = "labs/lab5/"
IMAGE = DIR + "cactus.jpg"
LAMBDA = 4

def calculate_mse_and_mae(original: np.ndarray, restored: np.ndarray) -> tuple[np.float64, np.float64]:
    mse = np.mean(np.square(np.subtract(original.astype(np.float64), restored.astype(np.float64))))
    mae = np.mean(np.abs(original.astype(np.float64) - restored.astype(np.float64)))

    return np.round(np.float64(mse), 2), np.round(np.float64(mae), 2)

cactus_img = cv2.imread(IMAGE)
cactus_img = scipy.ndimage.convolve(cactus_img)
cv2.imwrite(DIR + "splot.jpg", cactus_img)