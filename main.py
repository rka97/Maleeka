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

THRESH = 0.001

def main():
    img_pos = "images/capr6.png"
    image = io.imread(img_pos)
    image = preprocess(image)
    io.imsave('output.png', image * 255)
    image_lines = segment_lines(image)
    image_words = segment_words(image_lines)
    image_characters = segment_characters_latifa(image_words)
    ## Now need to segment words into characters.

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
    # threshold = threshold_otsu(grayscale_rotated)
    # thresholded_image = 1 * (grayscale_rotated < threshold)
    thresholded_image = 255 - grayscale_rotated
    # thresholded_image = thin(thresholded_image)
    show_image([thresholded_image])
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
            resized_line = resize(line, (40, int(line.shape[1] * (40 / line.shape[0]))), preserve_range=True, order=3, anti_aliasing=False)
            threshold = threshold_otsu(resized_line)
            resized_line = (resized_line > threshold) * 1
            io.imsave('images/lines/line' + str(len(lines)) +'.png', resized_line * 255)
            # resized_line = (resized_line > 1.5 * np.mean(resized_line)) * 1
            lines.append( resized_line )
            # show_image([resized_line])
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


# Implementation of the character segmentation algorithm from Latifa et al. (2004)
# The results were not very good.
def segment_characters_latifa(words):
    for word in words:
        # Find the horizontal junction line.
        horizontal_projection = np.sum(word, axis=1)
        junction_idx = np.argmax(horizontal_projection)
        # Find the top & bottom lines for each column.
        length, width = word.shape
        top_lines = np.argmax(word, axis=0)
        bottom_lines = length - 1 - np.argmax( np.flip(word, axis=0), axis=0)
        # Find the threshold.
        vertical_projection = np.sum(word, axis=0)
        threshold = mode(vertical_projection)[0][0]
        # Find the number of transitions in every column.
        num_transitions = np.sum(np.abs(word[0:length-2, :] - word[1:length-1, :]), axis=0)
        
        character_start = 0
        reading_characer = 0
        j = width - 1
        show_image([word])
        while j >= 0:
            column_vproj = vertical_projection[j]
            # print(column_vproj, threshold)
            if reading_characer == 0 and column_vproj > threshold:
                character_start = j
                reading_characer = 1
                print("Started a character!")
            elif reading_characer == 1:
                if (top_lines[j] <= junction_idx) and \
                    (bottom_lines[j] >= junction_idx) and \
                    (column_vproj <= threshold) and \
                    (num_transitions[j] == 2) and \
                    (bottom_lines[j] - top_lines[j] <= threshold) and \
                    (top_lines[j] >= top_lines[character_start]):
                        show_image([word[:, j:character_start+2]])
                        reading_characer = 0
                        print("ended a chracter!")
                elif j == 0:
                    show_image([word[:, j:character_start+2]])
            j -= 1



main()