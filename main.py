from matplotlib import cm
from matplotlib import pyplot as plt
import cv2
from imutils import grab_contours, contours
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
from scipy.signal import find_peaks
from skimage.filters.edges import convolve
from skimage.feature import canny
import cv2 as cv
from common import *
from preprocess import *
from segment_char import *
from generate_dataset import *
from os import listdir
from os.path import isfile, join


DATASET_DIRECTORY = 'dataset/'

def main():
	
	# TODO insert loop over all img_names here
	images = [f for f in listdir(DATASET_DIRECTORY + 'scanned/') if isfile(join(DATASET_DIRECTORY + 'scanned/', f))]
	for img_name in images:		
		print("filename", img_name)
		img_pos = DATASET_DIRECTORY + 'scanned/' + img_name
		file_pos = DATASET_DIRECTORY + 'text/' + img_name.split('.')[0] + '.txt'
		image = io.imread(img_pos)
		image = preprocess(image)
		io.imsave('output.png', image * 255)
		image_lines = segment_lines(image)
		image_words = segment_words(image_lines)
		words_characters = segment_characters_habd(image_words)

		# i = 0
		# for i in range(len(words_characters)):
		# 	debug_draw(image_words[i])
		# 	for char in words_characters[i]:
		# 		debug_draw(char)
		generate_dataset(words_characters, file_pos)


# TODO: Deprecate this.
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
		show_images([word])
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
						show_images([word[:, j:character_start+2]])
						reading_characer = 0
						print("ended a chracter!")
				elif j == 0:
					show_images([word[:, j:character_start+2]])
			j -= 1



main()
