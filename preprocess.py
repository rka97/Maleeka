import numpy as np
from skimage import io
from skimage.transform import rotate, resize, rescale
from skimage.color import rgb2gray
# from skimage.filters import threshold_otsu,threshold_minimum, gaussian
from skimage.filters import *
from skimage.morphology import *
from scipy.stats import mode
from scipy.signal import find_peaks
from skimage.filters.edges import convolve
from skimage.feature import canny
from common import *

# Do image preprocessing.
def preprocess(img):
    # deskew the image
    img = image_autocrop(img)
    grayscale = rgb2gray(img)
    angle = deskew.determine_skew(grayscale, sigma=2.0)
    # print(angle)
    if angle < -60:
        angle = 0
    grayscale_rotated = rotate(grayscale, angle, resize=True, mode='constant', cval=1) * 255 
    # threshold = threshold_otsu(grayscale_rotated)
    # thresholded_image = 1 * (grayscale_rotated < threshold)
    thresholded_image = 255 - grayscale_rotated
    # thresholded_image = thin(thresholded_image)
    return thresholded_image

# The goal of this is to segment the image into lines.
def segment_lines(img):
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
            line = img[lin_start:i, :]
            resized_line = resize(line, (LINE_HEIGHT, int(line.shape[1] * (LINE_HEIGHT / line.shape[0]))), preserve_range=True, order=3, anti_aliasing=False)
            # io.imsave('images/lines/line' + str(len(lines)) +'.png', resized_line * 255)
            # resized_line = (resized_line > 1.5 * np.mean(resized_line)) * 1
            if line.shape[0] > 5 and line.shape[1] > 5:
                lines.append( resized_line )
            # show_images([resized_line])
            reading_line = 0
    return lines

## TODO: merge this with the line segmentation algorithm.
##       or take the common parts into one function.
def segment_words(lines):
    words = []
    for line in lines:
        vertical_projection = np.sum( gaussian(line, sigma=1.6175), axis=0)
        threshold = np.max(vertical_projection) * 0.0495
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
                reading_word = 0
                threshold = 110
                word = (word > threshold) * 1
                word = image_autocrop(word)
                words.append( word )
            i -= 1
    return words
