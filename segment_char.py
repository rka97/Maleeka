
import cv2 as cv
from common import *
from preprocess import *
from segment_char_fixes import *

def detect_mid_chars(raw, word):
    word_skeleton = skeletonize(word)
    height, width = word.shape
    sigma_v = 3.5
    sigma_h = 2
    vertical_projection = np.sum(word, axis=0)
    horizontal_projection = np.sum(word_skeleton, axis=1)

    baseline = np.argmax(gaussian(horizontal_projection, sigma_h))
    hproj_highlight = np.zeros_like(word)
    hproj_highlight[baseline-BASELINE_THICKNESS+1:baseline+BASELINE_THICKNESS, :] = 1

    vproj_highlight = np.zeros_like(word)
    possible_splits = np.zeros(width, dtype=int)
    for w in range(width):
        above_bl = np.sum(word_skeleton[:baseline-BASELINE_THICKNESS+1, w])
        below_bl = np.sum(word_skeleton[baseline+BASELINE_THICKNESS:, w])
        avg_bl = np.mean(word_skeleton[baseline-BASELINE_THICKNESS+1:baseline+BASELINE_THICKNESS, w])
        if avg_bl > 0 and above_bl == 0 and below_bl == 0:
            possible_splits[w] = 1

    w = width-1
    while w >= 1:
        while (possible_splits[w] == 1 and possible_splits[w-1]  == 1):
            possible_splits[w] = 0
        w -= 1
    possible_splits[:3] = 0

    vproj_highlight[:, :] = possible_splits
    # debug_draw(raw, hproj_highlight, vproj_highlight)
    
    split_indices = np.flip(np.where(possible_splits == 1))[0]
    split_segments = get_segments_from_indices(word_skeleton, split_indices)
    split_indices = fix_segments(split_segments, split_indices)
    split_segments = get_segments_from_indices(raw, split_indices)

    vproj_highlight2 = np.zeros_like(vproj_highlight)
    vproj_highlight2[:, split_indices] = 1
    # debug_draw(skeletonize(raw), vproj_highlight, vproj_highlight2)

    # for segment in split_segments:
    #     debug_draw(segment)
    return split_segments

def get_segments_from_indices(word, split_indices):
    height, width = word.shape
    split_segments = []
    start = width-1
    for split_index in split_indices:
        segment = word[:, split_index:start+1]
        start = split_index
        split_segments.append(segment)
    split_segments.append(word[:, :start+1])
    return split_segments


def fill_dots(word_with_dots):
    word = word_with_dots.copy()
    _, contours, heirarchy = cv.findContours(word.astype('uint8')*255, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if (len(contours) != 0):
        is_hole = heirarchy [:,:,3] != -1
        is_hole = is_hole[0]
        is_secondary = np.array([contour.shape[0] <= 25 for contour in contours])
        #print("hole", len(is_hole),"secondary", len(is_secondary), "cnts", len(contours))
        for i, contour in enumerate(contours):
            if (not is_hole[i] and is_secondary[i]):
                    mask = np.ones_like(word).astype('uint8')
                    cv.drawContours(mask, [contour], 0, 0, -1)
                    word *= mask
    return word

def _segment_subword(raw, filled):
    # Isolated characters
    raw_cropped = image_autocrop(raw)
    sx, sy = raw_cropped.shape
    # print(sx, sy)
    if sy <= 13:
        print("Too small => Isolated..")
        # debug_draw(raw_cropped, raw_cropped)
        return [ raw ]
    # Middle/End Characters
    split_segments = detect_mid_chars(raw, filled)
    return split_segments

def sorting_key(subword):
    return subword[0]

def segment_characters_habd(words):
    words_characters = []
    for word in words:
        subwords = []
        filled_word = fill_dots(word)
        _, contours, heirarchy = cv.findContours(filled_word.astype('uint8')*255, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        cnt = 0
        for contour in contours:
            red_word = np.zeros_like(word)
            is_hole = (heirarchy[0][cnt][3] != -1)
            # print("is_hole: " + str(is_hole))
            min_width = np.inf
            max_width = -1
            for [point] in contour:
                if point[0] < min_width:
                    min_width = point[0]
                if point[0] > max_width:
                    max_width = point[0] 
                red_word[point[1], point[0]] = 1
            
            box_mask = np.zeros_like(word)
            box_mask[ :, min_width:max_width+1] = 1

            if not (is_hole):
                # debug_draw(word, box_mask, red_word)
                # print(max_width)
                subwords.append([ max_width, word[:, min_width:max_width+1], filled_word[:, min_width:max_width+1] ])
            # print("\n")
            cnt += 1
        
        subwords.sort(key=sorting_key, reverse=True)
        word_characters = []

        for [_, subword_pure, subword_filled] in subwords:
            subword_characters = _segment_subword(subword_pure, subword_filled)
            word_characters += subword_characters

        words_characters.append(word_characters)
    return words_characters