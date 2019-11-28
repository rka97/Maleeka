import numpy as np
import matplotlib.pyplot as plt
import deskew
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu


def show_image(images):
    for img in images:
        plt.imshow(img, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

# Do image preprocessing.
def preprocess(img):
    # deskew the image
    grayscale = rgb2gray(img)
    angle = deskew.determine_skew(grayscale)
    grayscale_rotated = rotate(grayscale, angle, resize=True, mode='constant', cval=1) * 255
    # threshold it to get a binarized image
    threshold = threshold_otsu(grayscale_rotated)
    thresholded_image = grayscale_rotated * (grayscale_rotated > threshold)
    return thresholded_image


img_pos = "images/capr6.png"
image = io.imread(img_pos)
output = preprocess(image)
io.imsave('output.png', output.astype(np.uint8))

# preprocess(img)