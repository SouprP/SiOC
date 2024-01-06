from PIL import Image, ImageEnhance

DIR = "latex/lab3/images/"
BAYER = DIR + "Bayer_Diff.jpg"
XTRANS = DIR + "X_Trans_Diff.jpg"
# Read the image
im = Image.open(XTRANS)

# Image brightness enhancer
enhancer = ImageEnhance.Brightness(im)

#factor = 1 #gives original image
#im_output = enhancer.enhance(factor)
#im_output.save(DIR + "Bayer_DIFF_br.jpg")

factor = 10 #brightens the image
im_output = enhancer.enhance(factor)
im_output.save(DIR + "XTRANS_DIFF_br.jpg")