import numpy as np
import matplotlib.pyplot as plt
import deskew
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray
# from skimage.filters import threshold_otsu,threshold_minimum, gaussian
from skimage.filters import *

THRESH = 0.001

def main():
    img_pos = "images/capr6.png"
    image = io.imread(img_pos)
    image = preprocess(image)
    io.imsave('output.png', image)
    image_lines = segment_lines(image)

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
    threshold = threshold_otsu(grayscale_rotated)
    thresholded_image = 1 * (grayscale_rotated < threshold)
    return thresholded_image


# The goal of this is to segment the image into lines.
def segment_lines(img):
    horizontal_projection = np.sum(img, axis=1) / 255
    horizontal_projection = gaussian(horizontal_projection, sigma=0.5) # smooth everything a little..
    threshold = np.max(horizontal_projection) * 0.01
    horizontal_projection = 1 * (horizontal_projection > threshold)
    lines = []
    lin_start = 0
    lin_end = -1
    reading_line = 0
    for i in range(horizontal_projection.size):
        line_value = horizontal_projection[i]
        if reading_line == 0 and line_value > 0:
            lin_start = i
            reading_line = 1
        elif reading_line == 1 and line_value <= 0:
            lin_end = i
            lines.append( img[lin_start:lin_end, :] )
            reading_line = 0
    show_image(lines)



main()