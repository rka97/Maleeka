
import numpy as np
from scipy.stats import mode
import os, re
from skimage.filters import *
from skimage.morphology import *
from skimage.measure import find_contours
import cv2 as cv
from util import *


# Implementation of "Printed Arabic Optical Character Segmentation" by Khader et al. (2015)
def segment_characters_khader(words):
    for word in words:
        # word = thin(word) * 1.0
        word = np.ascontiguousarray(word, dtype=np.uint8)
        im2, contours, hierarchy = cv.findContours(word, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        show_image([word])
        for contour in contours:
            curr_word = np.zeros(word.shape)
            if len(contour) > 30:
                for point in contour:
                    curr_word[point[0][1], point[0][0]] = 1
                    print(point[0][1], point[0][0])
                show_image([curr_word])
                print("\n\n")


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
                        word[:, j] = 1
                        word[:, character_start] = 1
                        # show_image([word[:, j:character_start+2]])
                        reading_characer = 0
                        print("ended a chracter!")
                elif j == 0 and reading_characer == 1:
                    word[:, j] = 1
                    word[:, character_start] = 1
                    # show_image([word[:, j:character_start+2]])
            j -= 1
        show_image([word])



### TODO: Segmentation by HMMs
def load_character_set(directory):
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            character_name = (re.search("[^_]*", filename)).group(0)
            character_pos = (re.search("([^_]*)(?:[.])", filename)).group(1)
            character_pos = character_pos[0]
            print(filename, character_name, character_pos)

