import numpy as np
from PIL import Image, ImageChops
import glob
import math

DIR = "labs/lab3/"
IMAGE = DIR + "Fella.jpg"

def cone(x, y):
    if not ((x|y)):
        return 10000
    return 1/math.sqrt(math.pow(x,2) + math.pow(y,2))

def equal(x,y):
    return 1

def maskToBit(image, mask, color):
    im = Image.new("1", size=image.size)
    px = im.load()
    for x in range(image.width):
        for y in range(image.height):
            px[x, y] = (mask[y % len(mask)][x % len(mask[0])] == color) #reversed order so easier creation of mask
    return im

def interpolation(image, bit, function):
    extendedImage = Image.new("RGB", size=tuple([image.width + 2 * radius, image.height + 2 * radius]), color=0)
    extendedImage.paste(image, tuple([radius, radius]))

    listResultBands = []
    listResultBandsPixels = []

    iValue = []
    iv = []

    iCount = []
    ic =[]

    for i in range(3):
        listResultBands.append(Image.new("L", size=tuple([image.width, image.height]), color=0))
        listResultBandsPixels.append(listResultBands[i].load())

        iValue.append(Image.new("F", size=tuple([image.width + 2 * radius, image.height + 2 * radius]), color=0))
        iv.append(iValue[i].load())

        iCount.append(Image.new("F", size=tuple([image.width + 2 * radius, image.height + 2 * radius]), color=0))
        ic.append(iCount[i].load())

        for w in range(-radius, radius + 1):
            for h in range(-radius, radius + 1):
                weigh = function(w, h)
                for x in range(radius, image.width + radius):
                    for y in range(radius, image.height + radius):
                        iv[i][x+w, y+h] += extendedImage.getpixel((x, y))[i] * weigh
                        ic[i][x+w, y+h] += bit[i].getpixel(tuple([x-radius, y - radius])) * weigh


        for x in range(radius, image.width + radius):
            for y in range(radius, image.height + radius):
                listResultBandsPixels[i][x - radius, y - radius] = int(iv[i][x, y] / ic[i][x, y])

    return Image.merge("RGB", tuple(listResultBands))

dIntToColors = {0: 'R',
                1: 'G',
                2: 'B'}

maskBayer = [['G', 'B'],
             ['R', 'G']]

maskXtrans = [['G', 'B', 'R', 'G', 'R', 'B'],
              ['R', 'G', 'G', 'B', 'G', 'G'],
              ['B', 'G', 'G', 'R', 'G', 'G'],
              ['G', 'R', 'B', 'G', 'B', 'R'],
              ['B', 'G', 'G', 'R', 'G', 'G'],
              ['R', 'G', 'G', 'B', 'G', 'G']]



mask = maskBayer
radius = 1
imgName = ["malpa.png", "kwiat.jpg"]
im = Image.open(IMAGE)
nm = Image.new("L", size=im.size, color=0)
a = []
masks = []

for i in range(3):
    masks.append(maskToBit(im, mask, dIntToColors[i]))
    split = im.split()[i]
    a.append((Image.composite(split, nm, masks[i])))


merged = Image.merge("RGB", tuple(a))
merged.show()
sth = interpolation(merged, masks, equal)
sth.show()
diff = ImageChops.difference(sth, im)
diff.show()
print("ednd")