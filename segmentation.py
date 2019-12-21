import numpy as np
import matplotlib.pyplot as plt
import deskew
from skimage import io
from skimage.transform import rotate, resize, rescale
from skimage.color import rgb2gray
# from skimage.filters import threshold_otsu,threshold_minimum, gaussian
from skimage.filters import *
from skimage.morphology import *
from scipy.stats import mode
from util import *

THRESH = 0.001
LINE_HEIGHT = 60

# Do image preprocessing.
def preprocess(img, fix_skew=True):
    # show_image([img])
    # deskew the image
    grayscale = rgb2gray(img)
    # show_image([img])
    if fix_skew:
        angle = deskew.determine_skew(grayscale)
        grayscale = rotate(grayscale, angle, resize=True, mode='constant', cval=1)
    grayscale = 255 * (grayscale == 0)
    # threshold = threshold_otsu(grayscale_rotated)
    # thresholded_image = 1 * (grayscale_rotated < threshold)
    # thresholded_image = thin(thresholded_image)
    # show_image([thresholded_image])
    return grayscale

def autocrop_line(line):
    vertical_projection = np.sum(line, axis=0)
    right = vertical_projection.size - 1
    while right >= 0 and vertical_projection[right] == 0:
        right -= 1
    left = 0
    while left < vertical_projection.size and vertical_projection[left] == 0:
        left += 1
    return line[:, left:right]
    

# The goal of this is to segment the image into lines.
def segment_lines(img, character_mode=False):
    horizontal_projection = np.sum(img, axis=1) / 255
    horizontal_projection = gaussian(horizontal_projection, sigma=0.5) # smooth everything a little..
    threshold = np.max(horizontal_projection) * 0.01
    horizontal_projection = 1 * (horizontal_projection > threshold)
    lines = []
    lin_start = 0
    reading_line = 0
    for i in range(horizontal_projection.size):
        line_value = horizontal_projection[i]
        if reading_line == 0 and line_value > 0:
            lin_start = i
            reading_line = 1
        elif reading_line == 1 and line_value <= 0:
            if character_mode is True:
                line = img[lin_start:, :]
            else:
                line = img[lin_start:i, :]
            resized_line = resize(line, (LINE_HEIGHT, int(line.shape[1] * (LINE_HEIGHT / line.shape[0]))), preserve_range=True, order=3, anti_aliasing=False)
            threshold = threshold_otsu(resized_line)
            resized_line = (resized_line > threshold) * 1
            # io.imsave('images/lines/line' + str(len(lines)) +'.png', resized_line * 255)
            # resized_line = (resized_line > 1.5 * np.mean(resized_line)) * 1
            lines.append( autocrop_line(resized_line) )
            reading_line = 0
        
    return lines

## TODO: merge this with the line segmentation algorithm.
##       or take the common parts into one function.
def segment_words(lines):
    words = []
    for line in lines:
        vertical_projection = np.sum(line, axis=0)
        vertical_projection = gaussian(vertical_projection, sigma=1.5)
        threshold = np.max(vertical_projection) * 0.1
        vertical_projection = 1 * (vertical_projection > threshold)
        word_start = 0
        reading_word = 0
        i = vertical_projection.size - 1
        while i >= 0:
            line_value = vertical_projection[i]
            if reading_word == 0 and line_value > 0:
                word_start = i
                reading_word = 1
            elif reading_word == 1 and line_value <= 0:
                word = line[:, i+1:word_start+1]
                words.append( word )
                reading_word = 0
                # show_image([word])
            i -= 1
    return words