from main import resize_nearest, resize_bilinear, resize_bicubic, rotate
import cv2
import numpy as np
import threading
from t import rotate_nearest_neighbor, rotate_bilinear, rotate_bicubic


# Orginalny obraz
IMG : np.ndarray = cv2.imread("labs/lab2/tests/monke_512.jpg")
H_ORG, W_ORG = IMG.shape[:2]

IMG_NEW_3nn = np.ndarray = cv2.imread("latex/lab2/images/3x_BACK_TO_ORG_nn.jpg")
IMG_NEW_3bc = np.ndarray = cv2.imread("latex/lab2/images/3x_BACK_TO_ORG_bc.jpg")
IMG_NEW_3bl = np.ndarray = cv2.imread("latex/lab2/images/3x_BACK_TO_ORG_bl.jpg")
IMG_NEW_5nn = np.ndarray = cv2.imread("latex/lab2/images/5x_BACK_TO_ORG_nn.jpg")
IMG_NEW_5bc = np.ndarray = cv2.imread("latex/lab2/images/5x_BACK_TO_ORG_bc.jpg")
IMG_NEW_5bl = np.ndarray = cv2.imread("latex/lab2/images/5x_BACK_TO_ORG_bl.jpg")
def error_calculator(img1, img2):
    #return np.average(abs(np.array(img1, dtype="int16") -
                          #np.array(img2, dtype="int16")))
    return np.square(np.subtract(img1, img2)).mean()

# Test 1
def resize_test_5x():
    # # POWIĘSZENIE 5X10%
    RES_IMG = IMG
    for i in range(5):
        H, W = RES_IMG.shape[:2]
        RES_IMG = resize_bicubic(RES_IMG, int(1.1*H), int(1.1*W))
    cv2.imwrite("labs/lab2/tests/5x_10%_nn.jpg", RES_IMG)
    RES_IMG = resize_bicubic(RES_IMG, H_ORG, W_ORG)
    
    # POMNIEJSZENIE DO ORGINALNEGO ROZMIAR
    #RES_IMG = resize(RES_IMG, H_ORG, W_ORG)
    #cv2.imwrite("labs/lab2/tests/5x_BACK_TO_ORG_nn.jpg", RES_IMG)

def resize_test_3x():
     # # POMNIEJSZENIE 3X10%
    RES_IMG = IMG
    for i in range(3):
        H, W = RES_IMG.shape[:2]
        RES_IMG = resize_bicubic(RES_IMG, int(0.9*H), int(0.9*W))
    cv2.imwrite("labs/lab2/tests/3x_10%_bc.jpg", RES_IMG)
    RES_IMG = resize_bicubic(RES_IMG, H_ORG, W_ORG)
    
    # POMNIEJSZENIE DO ORGINALNEGO ROZMIAR
    #RES_IMG = resize(RES_IMG, H_ORG, W_ORG)
    #cv2.imwrite("labs/lab2/tests/3x_BACK_TO_ORG_bc.jpg", RES_IMG)
# Test 2
def rotate_test():
    RES_IMG = IMG
    #for i in range(1, int(360/12) + 1):
    for i in range(1, 4 + 1):
        print(i)
        RES_IMG = rotate(RES_IMG, 12*i)

    cv2.imwrite("labs/lab2/tests/rotate_nn.jpg", RES_IMG)
    print(error_calculator(IMG, RES_IMG))
#threading.Thread(target=resize_test_5x).start()
#threading.Thread(target=resize_test_3x).start()

rotate_test()
#img_r = rotate_nearest_neighbor(IMG, 12*4)
#cv2.imwrite("labs/lab2/tests/rotate_12_4.jpg", img_r)
#cv2.imshow("img_r", img_r)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#print(error_calculator(IMG, IMG_NEW_3nn), ",3nn")
#print(error_calculator(IMG, IMG_NEW_3bc), ",3bc")
#print(error_calculator(IMG, IMG_NEW_3bl), ",3bl")
#print(error_calculator(IMG, IMG_NEW_5nn), ",5nn")
#print(error_calculator(IMG, IMG_NEW_5bc), ",5bc")
#print(error_calculator(IMG, IMG_NEW_5bl), ",5bl")

#resize_test_5x()
#resize_text_3x()
#resized_near = resize_nearest(IMG, int(H_ORG*1.1), int(W_ORG*1.1))
#resized_biline = resize_bilinear(IMG, int(H_ORG*1.1), int(W_ORG*1.1))
#resized_bicub = resize_bicubic(IMG, int(H_ORG*1.1), int(W_ORG*1.1))

#resized_bicub = resize_bicubic(IMG, int(0.2*H_ORG), int(0.2*W_ORG))
#cv2.imwrite("labs/lab2/tests/monke_512_bicub.jpg", resized_bicub)
#cv2.imwrite("labs/lab2/tests/monke_512_biline.jpg", resized_biline)
#cv2.imwrite("labs/lab2/tests/monke_512_near.jpg", resized_near)
