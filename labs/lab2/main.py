import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
from func_timer import timer

plot, axis = plt.subplots(1, 2)

def resize(img : np.ndarray, new_h, new_w):
    old_h, old_w, channels = img.shape
    resized = np.zeros((new_h, new_w, channels))

    w_scale_factor = old_w / new_w if new_h != 0 else 0
    h_scale_factor = old_h / new_h if new_w != 0 else 0

    for i in range(new_h):
        for j in range(new_w):
            x = i * h_scale_factor
            y = j * w_scale_factor
            x_floor = math.floor(x)
            y_floor = math.floor(y)
            x_ceil = min(old_h - 1, math.ceil(x))
            y_ceil = min(old_w - 1, math.ceil(y))

            if x_ceil == x_floor and y_ceil == y_floor:
                resized[i, j, :] = img[x_floor, y_floor, :]
            elif x_ceil == x_floor:
                q1 = img[x_floor, y_floor, :]
                q2 = img[x_floor, y_ceil, :]
                resized[i, j, :] = q1 * (y_ceil - y) + q2 * (y - y_floor)
            elif y_ceil == y_floor:
                q1 = img[x_floor, y_floor, :]
                q2 = img[x_ceil, y_floor, :]
                resized[i, j, :] = q1 * (x_ceil - x) + q2 * (x - x_floor)
            else:
                v1 = img[x_floor, y_floor, :]
                v2 = img[x_ceil, y_floor, :]
                v3 = img[x_floor, y_ceil, :]
                v4 = img[x_ceil, y_ceil, :]
                q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                resized[i, j, :] = q1 * (y_ceil - y) + q2 * (y - y_floor)
    
    return resized
    
@timer
def resize_nearest(img, new_h, new_w):
    old_h, old_w, channels = img.shape
    resized = np.zeros((new_h, new_w, channels))

    #enlarge_time = math.sqrt((new_h * new_h) / (new_h*new_h))

    for i in range(new_h):
        for j in range(new_w):
            x = int(i * old_h / new_h)
            y = int(j * old_w / new_w)
            resized[i, j, :] = img[x, y, :]
    
    return resized

@timer
def resize_bilinear(img, new_h, new_w):
    old_h, old_w, channels = img.shape
    resized = np.zeros((new_h, new_w, channels))

    scale_x = old_w / new_w
    scale_y = old_h / new_h

    # 3
    for k in range(3):
        for i in range(new_h):
            for j in range(new_w):
                x = (j+0.5) * (scale_x) - 0.5
                y = (i+0.5) * (scale_y) - 0.5

                x_int = int(x)
                y_int = int(y)

                # Prevent crossing
                x_int = min(x_int, old_w-2)
                y_int = min(y_int, old_h-2)

                x_diff = x - x_int
                y_diff = y - y_int

                a = img[y_int, x_int, k]
                b = img[y_int, x_int+1, k]
                c = img[y_int+1, x_int, k]
                d = img[y_int+1, x_int+1, k]

                pixel = a*(1-x_diff)*(1-y_diff) + b*(x_diff) * \
                    (1-y_diff) + c*(1-x_diff) * (y_diff) + d*x_diff*y_diff

                resized[i, j, k] = pixel.astype(np.uint8)
    return resized

def weight(x):
    a = -0.5
    pos_x = abs(x)
    if -1 <= abs(x) <= 1:
        return ((a+2)*(pos_x**3)) - ((a+3)*(pos_x**2)) + 1
    elif 1 < abs(x) < 2 or -2 < x < -1:
        return ((a * (pos_x**3)) - (5*a*(pos_x**2)) + (8 * a * pos_x) - 4*a)
    else:
        return 0
    
@timer
def resize_bicubic(img, new_h, new_w):
    old_h, old_w, channels = img.shape
    resized = np.zeros((new_h, new_w, channels))

    # for c in range(channels):
    for c in range(img.shape[2]):
        for i in range(new_h):
            for j in range(new_w):
                xm = (i + 0.5) * (old_h/new_h) - 0.5
                ym = (j + 0.5) * (old_w/new_w) - 0.5

                xi = math.floor(xm)
                yi = math.floor(ym)

                u = xm - xi
                v = ym - yi

                out = 0
                for n in range(-1, 3):
                    for m in range(-1, 3):
                        if ((xi + n < 0) or (xi + n >= old_h) or (yi + m < 0) or (yi + m >= old_w)):
                            continue

                        out += (img[xi+n, yi+m, c] * (weight(u - n) * weight(v - m)))

                resized[i, j, c] = np.clip(out, 0, 255)

    return resized

@timer
def rotation(angle, flags=cv2.INTER_NEAREST):
    img = cv2.imread('skull.png')
    # convert of color because OpenCV works in BGR and Matplotlib in RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width = img.shape[:2]
    center = (int(height / 2), int(width / 2))

    rotationMatrix = cv2.getRotationMatrix2D(
        center=center, angle=angle, scale=1.0)  # create rotation matrix
    '''
    rotationMatrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    '''

    start = time.time()
    rotatedImage = cv2.warpAffine(src=img, M=rotationMatrix, dsize=(
        height, width), flags=flags)  # making transformation
    end = time.time()
    print(end - start)
    plt.imsave('skull_rotated.png', rotatedImage)

    rotationMatrix2 = cv2.getRotationMatrix2D(
        center=center, angle=-angle, scale=1.0)
    backToNormal = cv2.warpAffine(
        src=rotatedImage, M=rotationMatrix2, dsize=(height, width), flags=flags)
    plt.imsave('backToNormal_rotated.png', backToNormal)

@timer
def rotation_all_cv(img, angle, flags=cv2.INTER_LINEAR):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width = img.shape[:2]
    center = (int(height / 2), int(width / 2))

    rotationMatrix = cv2.getRotationMatrix2D(center=center, 
        angle=angle, scale=0.5)
        
    rotatedIMG = cv2.warpAffine(src=img, M=rotationMatrix,
        dsize=(height, width), flags=flags)

    return rotatedIMG

#image : np.ndarray = cv2.imread("labs/lab2/tests/Pasikonik.jpg")
#resized = resize_bicubic(image, 3000, 1750)
#cv2.imwrite("labs/lab2/pasikonik_resized.jpg", resized)

#rotated45 = rotate(image, 45)
#cv2.imwrite("lab2/pasikonik_rotated45.jpg", rotated45)