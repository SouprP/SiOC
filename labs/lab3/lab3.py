import cv2
import numpy as np
from func_timer import timer

DIR = "labs/lab3/"
IMAGE = DIR + "Fella.jpg"

BAYER_MASK = [
    ['G', 'R'],
    ['B', 'G']
]

XTRANS_MASK = [
    ['G', 'B', 'R', 'G', 'R', 'B'],
    ['R', 'G', 'G', 'B', 'G', 'G'],
    ['B', 'G', 'G', 'R', 'G', 'G'],
    ['G', 'R', 'B', 'G', 'B', 'R'],
    ['B', 'G', 'G', 'B', 'G', 'G']
]

def error_calculator(img1, img2):
    #return np.average(abs(np.array(img1, dtype="int16") -
                          #np.array(img2, dtype="int16")))
    return np.square(np.subtract(img1, img2)).mean()

@timer
def bayer(image):
    #mat = np.zeros((image.shape[0], image.shape[1]))
    mat = np.array(image)
    print(mat)

    mat[::2, ::2] = mat[::2, ::2] * [0, 1, 0]
    mat[::2, 1::2] = mat[::2, 1::2] * [0, 0, 1]
    mat[1::2, ::2] = mat[1::2, ::2] * [1, 0, 0]
    mat[1::2, 1::2] = mat[1::2, 1::2] * [0, 1, 0]
    cv2.imwrite(DIR + "bayer.bmp", mat)
    return mat

@timer
def XTrans(img):
    mat = np.array(img)
    
    for h in range(img.shape[0]): 
        for w in range(img.shape[1]): # iterating through each pixel
            if h%6 == 0:
                if str(w%6) in '03':
                    img[h][w] = [0 ,img[h][w][1] ,0]
                elif str(w%6) in '15':
                    img[h][w] = [img[h][w][0],0 ,0]
                else:
                    img[h][w] = [0, 0, img[h][w][2]]
            elif str(h%6) in '15':
                if str(w%6) in '1245':
                    img[h][w] = [0 ,img[h][w][1] ,0]
                elif str(w%6) in '3':
                    img[h][w] = [img[h][w][0],0 ,0]
                else:
                    img[h][w] = [0, 0, img[h][w][2]]
            elif str(h%6) in '24':
                if str(w%6) in '1245':
                    img[h][w] = [0 ,img[h][w][1] ,0]
                elif str(w%6) in '0':
                    img[h][w] = [img[h][w][0],0 ,0]
                else:
                    img[h][w] = [0, 0, img[h][w][2]]
            elif h%6 == 3:
                if str(w%6) in '03':
                    img[h][w] = [0 ,img[h][w][1] ,0]
                elif str(w%6) in '24':
                    img[h][w] = [img[h][w][0],0 ,0]
                else:
                    img[h][w] = [0, 0, img[h][w][2]]
    cv2.imwrite(DIR + "X_Trans.bmp", img)
    return img  

ORG_IMG = cv2.imread(IMAGE)
BAYER_IMG = bayer(cv2.imread(IMAGE))
XTRANS_IMG = XTrans(cv2.imread(IMAGE))


print("Bayer: ", error_calculator(ORG_IMG, BAYER_IMG))
print("XTrans: ", error_calculator(ORG_IMG, XTRANS_IMG))
print("Bayer - XTrans: ", error_calculator(BAYER_IMG, XTRANS_IMG))