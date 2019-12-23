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
<<<<<<< HEAD
from skimage.morphology import skeletonize
from scipy.signal import argrelextrema


THRESH = 0.001
=======
from scipy.signal import find_peaks
from skimage.filters.edges import convolve
from skimage.feature import canny
import cv2 as cv
from common import *
from preprocess import *
from segment_char import *
>>>>>>> 6a1f4e8b36354201a062bb3febeaddbaac0b7996


def main():
<<<<<<< HEAD
	img_pos = "images/capr6.png"
	image = io.imread(img_pos)
	image = preprocess(image)
	io.imsave('output.png', image * 255)
	image_lines = segment_lines(image)
	image_words, median_locs= segment_words(image_lines)
	print("words length", len(image_words), "median length", len(median_locs))
	# show_image(image_words)
	image_characters = segment_characters_sari(image_words, median_locs)
	# image_characters = segment_characters_latifa(image_words)
	# Now need to segment words into characters.


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
	grayscale_rotated = rotate(
		grayscale, angle, resize=True, mode='constant', cval=1) * 255
	# threshold = threshold_otsu(grayscale_rotated)
	# thresholded_image = 1 * (grayscale_rotated < threshold)
	thresholded_image = 255 - grayscale_rotated
	# thresholded_image = thin(thresholded_image)
	# show_image([thresholded_image])
	return thresholded_image


# The goal of this is to segment the image into lines.
def segment_lines(img):
	horizontal_projection = np.sum(img, axis=1) / 255
	# smooth everything a little..
	horizontal_projection = gaussian(horizontal_projection, sigma=0.5)
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
			resized_line = resize(line, (40, int(
				line.shape[1] * (40 / line.shape[0]))), preserve_range=True, order=3, anti_aliasing=False)
			threshold = threshold_otsu(resized_line)
			resized_line = (resized_line > threshold) * 1
			io.imsave('images/lines/line' + str(len(lines)) +
					  '.png', resized_line * 255)
			# resized_line = (resized_line > 1.5 * np.mean(resized_line)) * 1
			lines.append(resized_line)
			# show_image([resized_line])
			reading_line = 0
	return lines

# TODO: merge this with the line segmentation algorithm.
# or take the common parts into one function.


def segment_words(lines):
	words = []
	medians = []
	for line in lines:
		horizontal_projection = np.sum(line, axis=1)
		max_idx = argrelextrema(horizontal_projection, np.greater)
		if len(max_idx[0]) < 2:
			print("disaster", max_idx[0])
			print("histogram", horizontal_projection)
		max_values = horizontal_projection[max_idx]
		max_max_idx = max_values.argsort()[-2:][::-1]
		[median_start, median_end] = np.sort(np.array(max_idx)[0][max_max_idx])
		print("histogram", horizontal_projection)
		print("max_idx", max_idx)
		print("median", median_start, median_end)

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
				words.append(word)
				medians.append(tuple([median_start, median_end]))
				reading_word = 0
				# show_image([word])
			i -= 1
	return words, medians
=======
    img_pos = "images/scanned/capr4.png"
    image = io.imread(img_pos)
    image = preprocess(image)
    io.imsave('output.png', image * 255)
    image_lines = segment_lines(image)
    image_words = segment_words(image_lines)
    words_characters = segment_characters_habd(image_words)

    i = 0
    for i in range(len(words_characters)):
        debug_draw(image_words[i])
        for char in words_characters[i]:
            debug_draw(char)
    # TODO: classify characters.
>>>>>>> 6a1f4e8b36354201a062bb3febeaddbaac0b7996


# TODO: Deprecate this.
# Implementation of the character segmentation algorithm from Latifa et al. (2004)
# The results were not very good.
def segment_characters_latifa(words):
<<<<<<< HEAD
	for word in words:
		# Find the horizontal junction line.
		horizontal_projection = np.sum(word, axis=1)
		junction_idx = np.argmax(horizontal_projection)
		# Find the top & bottom lines for each column.
		length, width = word.shape
		top_lines = np.argmax(word, axis=0)
		bottom_lines = length - 1 - np.argmax(np.flip(word, axis=0), axis=0)
		# Find the threshold.
		vertical_projection = np.sum(word, axis=0)
		threshold = mode(vertical_projection)[0][0]
		# Find the number of transitions in every column.
		num_transitions = np.sum(
			np.abs(word[0:length-2, :] - word[1:length-1, :]), axis=0)

		character_start = 0
		reading_characer = 0
		j = width - 1
		# show_image([word])
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


# Implementation of the contour tracing character segmentation algorithm from Sari et al. (2002)

def segment_characters_sari(words, median_locs):
	for word, median in zip(words, median_locs):
		original_word = word.copy()
		word = skeletonize(word)
		show_image([word, original_word])
		word = word * 255
		word = word.astype('uint8')
		# get the contours of each symbol
		cnts, hierarchy = cv2.findContours(
			word.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		# mark holes in the contour
		is_hole = hierarchy[:,:, 3] != -1
		is_hole = is_hole[0]
		# marking secondary contours (useful for later AR rules)
		is_secondary = np.array([cnt.shape[0] <= 20 for cnt in cnts])
		# getting contours centers
		# M = [cv2.moments(cnt) for cnt in cnts]
		cnts_centers = [(int(np.mean(cnt[:,:,0])),int(np.mean(cnt[:,:,1]))) for cnt in cnts]
		# cnts = grab_contours(cnts)
		# sort contours from left to right (the correct order of symbols)
		# and do so with all their relevant data
		cnts, _ = zip(*sorted(zip(cnts, cnts_centers),
												key=lambda b: b[1][0], reverse=True))
		cnts = list(cnts)
		is_hole, _ = zip(*sorted(zip(is_hole, cnts_centers),
												key=lambda b: b[1][0], reverse=True))
		is_hole = np.array(is_hole)
		is_secondary, _ = zip(*sorted(zip(is_secondary, cnts_centers),
												key=lambda b: b[1][0], reverse=True))
		is_secondary = np.array(is_secondary)
		cnts_centers = sorted(cnts_centers,
												key=lambda b: b[0], reverse=True)
		# cnts, s = contours.sort_contours(cnts, method='right-to-left')
		# refine contours to keep just the lower bit of each one
		refined_cnts = []
		for cnt in cnts:
			refined_cnt = []
			for x in np.unique(cnt[:,:,0][:,0]):
				idx = np.where(cnt[:,:,0] == x)
				max_val = cnt[idx][:,1].max()
				refined_cnt.append([[x, max_val]])
			refined_cnts.append(np.array(refined_cnt))
		# refined_cnts = tuple(refined_cnts)

		# get first derivative of refined contours
		dy = [np.diff(cnt[:,:,1][:, 0]) for cnt in refined_cnts]

		local_minima = []
		consider_for_local_minima = np.invert(is_hole | is_secondary) 
		for j in range(len(dy)):
			if consider_for_local_minima[j]:
				d = dy[j]
			else:
				continue
			
			d_sign = np.sign(d)
			if len(d_sign) == 0:
				continue
			last_sign_idx = (d_sign != 0).argmax()
			last_sign = d_sign[last_sign_idx]
			if last_sign == 0:
				continue

			for i in range(last_sign_idx, len(d_sign)):
				if d_sign[i] == last_sign:
					last_sign = d_sign[i]
					last_sign_idx = i
				if d_sign[i] * last_sign == -1:
					if d_sign[i] == -1 and last_sign == 1:
						local_minima.append(refined_cnts[j][:,:,0][:,0][last_sign_idx])
						# print("Appended:", refined_cnts[j][:,:,0][:,0][last_sign_idx])
					last_sign = d_sign[i]
					last_sign_idx = i
		print(local_minima)
		for x in local_minima:
			original_word[:, x] = 1
		show_image([original_word])





		# Find the horizontal junction line.
		# horizontal_projection = np.sum(word/255, axis=1)
		# max_idx = argrelextrema(horizontal_projection, np.greater)
		# if len(max_idx[0]) < 2:
		# 	print("disaster", max_idx[0])
		# 	print("histogram", horizontal_projection)
		# max_values = horizontal_projection[max_idx]
		# max_max_idx = max_values.argsort()[-2:][::-1]
		# [median_end, median_start] = np.array(max_idx)[0][max_max_idx]



		# iterate over local minimas to check for validity
  
# Rule 5
def follows_hole(x, cnts, refined_cnts, is_hole, is_secondary):
	if cuts_hole(x):
		return False
	distance_from_hole_threshold = 10
	for cnt, hole in zip(refined_cnts, is_hole):
		if hole:
			hole_region = cnt[:,:,0][:,0]
			if x < hole_region[0] and hole_region - x < distance_from_hole_threshold:
				return True
	return False

# Rule 7
def cuts_hole(x, cnts, refined_cnts, is_hole, is_secondary):
	for cnt, hole in zip(refined_cnts, is_hole):
		if hole:
			hole_region = cnt[:,:,0][:,0]
			if x > hole_region[0] and x < hole_region[-1]:
				return True
	return False

# Rule 5
def follows_hole(x, cnts, refined_cnts, is_hole, is_secondary):
	distance_from_hole_threshold = 10
	for cnt, hole in zip(refined_cnts, is_hole):
		if hole:
			hole_region = cnt[:,:,0][:,0]
			if x < hole_region[0] and hole_region[0] - x < distance_from_hole_threshold:
				return True
	return False


def follows_descender(x, cnts, refined_cnts, is_hole, is_secondary, median):
	distance_from_descender_threshold = 10
	for cnt, refined_cnt, main_cnt in zip(cnts, refined_cnts, np.invert(is_hole | is_secondary)):
		if main_cnt:
			refined_cnt_region = refined_cnt[:,:,0][:,0]
			if x > refined_cnt_region[-1]:
				continue
			# for 
			# if refined_cnt_region[] - x < distance_from_hole_threshold:
			# 	return True
	return False


		
main()
=======
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
>>>>>>> 6a1f4e8b36354201a062bb3febeaddbaac0b7996
