import cv2
import numpy as np
from PIL import Image
from func_timer import timer

DIR = "labs/lab3/"
IMAGE = DIR + "Fella.jpg"

@timer
def Mosaic_Bayer(img):
    #[0, 1, 0] -> green, [1, 0, 0] -> blue, [0, 0, 1] -> red     
    n = np.array(img)
    
    n[::2, ::2] = n[::2, ::2] * [0, 1, 0]
    n[::2, 1::2]  = n[::2, 1::2] * [0, 0, 1]
    n[1::2, ::2] = n[1::2, ::2] * [1, 0, 0]
    n[1::2, 1::2]  = n[1::2, 1::2] * [0, 1, 0]
    
    cv2.imwrite("Bayer.jpg", n)    

@timer          
def Mosaic_Xtrans(img):
    #img[h][w] = [0 ,img[h][w][1] ,0] -> green, [img[h][w][0],0 ,0] -> blue, [0, 0, img[h][w][2]] -> red     
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
    cv2.imwrite("X_Trans.jpg", img)  
    

@timer
def Demosaic(image, mask):
    img = np.array(image)
    R, G, B = np.array([cv2.filter2D(img[:,:,n], -1, mask[n]) for n in range(3)])
    stack = np.dstack((R, G, B))
    result = Image.fromarray(stack)

    return result

original = cv2.cvtColor(cv2.imread(IMAGE), cv2.COLOR_BGR2RGB)
bayer_kernel = np.array([np.ones((2, 2)) * w for w in [1, 1/2, 1]])

xtrans_mask = np.array([[0   , 0   , 0   , 0   , 0   , 0],
                        [0   , 0.25, 0.5 , 0.5 , 0.25, 0],
                        [0   , 0.5 , 1.  , 1.  , 0.5 , 0],
                        [0   , 0.5 , 1.  , 1.  , 0.5 , 0],
                        [0   , 0.25, 0.5 , 0.5 , 0.25, 0],
                        [0   , 0   , 0   , 0   , 0   , 0]])
xtrans_kernel = np.array([xtrans_mask * w for w in [1/2, 1/5, 1/2]])

Mosaic_Bayer(original)
Mosaic_Xtrans(original)

Demosaic(cv2.imread("./Bayer.jpg"), bayer_kernel).save('Bayer_Demo.jpg')
Demosaic(cv2.imread("./X_Trans.jpg"), xtrans_kernel).save('X_Trans_Demo.jpg')

Image.fromarray(cv2.subtract(original, cv2.imread("./Bayer_Demo.jpg"))*2).save('Bayer_Diff.jpg')
Image.fromarray(cv2.subtract(original, cv2.imread("./X_Trans_Demo.jpg"))*2).save('X_Trans_Diff.jpg')

Bayer_ALL = np.concatenate((cv2.imread("./Bayer.jpg"), cv2.imread("./Bayer_Demo.jpg"), cv2.imread("./Bayer_Diff.jpg")), axis=1)
X_Trans_ALL = np.concatenate((cv2.imread("./X_Trans.jpg"), cv2.imread("./X_Trans_Demo.jpg"), cv2.imread("./X_Trans_Diff.jpg")), axis=1)

cv2.imwrite('Bayer_ALL.jpg', Bayer_ALL)
cv2.imwrite('X_Trans_ALL.jpg', X_Trans_ALL)

print('Bayer diff: ', np.square(np.subtract(cv2.imread(IMAGE), cv2.imread("./Bayer_Demo.jpg"))).mean())
print('X-Trans diff: ', np.square(np.subtract(cv2.imread(IMAGE), cv2.imread("./X_Trans_Demo.jpg"))).mean())

cv2.waitKey(0)
cv2.destroyAllWindows()