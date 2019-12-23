
import cv2 as cv
from common import *
from preprocess import *

# check if the segment has a hole
def has_hole(word):
    _, contours, heirarchy = cv.findContours(word.astype('uint8')*255, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if (len(contours) == 0):
        return False
    is_hole = heirarchy [:,:,3] != -1
    is_hole = is_hole[0]
    return np.max(is_hole) > 0

def detect_mid_chars(raw, word):
    BASELINE_THICKNESS = 2
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
    # debug_draw(word_skeleton, vproj_highlight, hproj_highlight)

    split_indices = np.flip(np.where(possible_splits == 1))[0]
    split_segments = get_segments_from_indices(word_skeleton, split_indices)
    split_indices = fix_segments(split_segments, split_indices)
    split_segments = get_segments_from_indices(raw, split_indices)
    # for segment in split_segments:
    #     debug_draw(segment)
    return split_segments


def get_split_index(seg_cntr, max_seg):
    if seg_cntr == max_seg-1:
        return max_seg-2
    else:
        return seg_cntr

# TODO: FIX ALL THE SPLITS.
    # RULES:
    # 1. SAD/DAD:
    #    If I have a hole, the next has low width, and am much higher than the next.
def fix_segments(segments, split_indices):
    seg_cntr = 0
    n_deleted = 0
    n_seg = len(segments)
    mask = np.ones_like(split_indices, dtype=bool)
    while seg_cntr < n_seg:
        curr_segment = image_autocrop(segments[seg_cntr])
        # debug_draw(curr_segment)
        curr_sx, curr_sy = curr_segment.shape
        print("Current info: ", curr_sx, curr_sy)

        if curr_sx <= 2 or curr_sy <= 2:
            mask[get_split_index(seg_cntr, n_seg)] = False
            seg_cntr += 2
            continue
        if seg_cntr < (n_seg - 1):
            next_segment = image_autocrop(segments[seg_cntr + 1])
            next_sx, next_sy = next_segment.shape
            print("Next info: ", next_sx, next_sy)
            if has_hole(curr_segment) and not has_hole(next_segment):
                # Candidate to be a SAD/DAD
                if (curr_sy > 2 * next_sy and curr_sx > next_sx and next_sx < next_sy - 1):
                    mask[get_split_index(seg_cntr, n_seg)] = False
                    seg_cntr += 2
                    print("Over-segmented SAD detected.")
                    continue
                    # debug_draw(curr_segment)
                    # debug_draw(next_segment)
            if next_sx > 2 * next_sy and curr_sx >= curr_sy and not(curr_sx >= 2 * curr_sy):
                # Candidate to be an over-segmented noon.
                horizontal_projection = np.sum(segments[seg_cntr], axis=1)
                baseline = np.argmax( horizontal_projection )
                print("baseline: ", baseline)
                if (baseline >= 18):
                    print("Over-segmented NUN detected.")
                    mask[get_split_index(seg_cntr, n_seg)] = False
                    seg_cntr += 2
                    continue
            if seg_cntr < (n_seg - 2):
                next2_segment = image_autocrop(segments[seg_cntr + 2])
                next2_sx, next2_sy = next2_segment.shape
                print("next2 info: ", next2_sx, next2_sy)
                # Candidate to be a SEEN/SHEEN
                hp1 = np.sum(segments[seg_cntr], axis=1)
                bl1 = np.argmax(hp1)
                hp2 = np.sum(segments[seg_cntr+1], axis=1)
                bl2 = np.argmax(hp2)
                hp3 = np.sum(segments[seg_cntr+2], axis=1)
                bl3 = np.argmax(hp3)
                print(np.mean(hp1), bl1, np.mean(hp2), bl2, np.mean(hp3), bl3)
                if (bl1 == bl2 and bl2 == bl3 and np.mean(hp1) <= 0.6 and np.mean(hp2) <= 0.6 and np.mean(hp3) <= 0.6):
                    print("Over-segmented SEEN detected.")
                    mask[get_split_index(seg_cntr, n_seg)] = False
                    mask[get_split_index(seg_cntr+1, n_seg)] = False
                    seg_cntr += 3
                    continue
        print("\n")
        seg_cntr += 1
    print("fix_segments: END.")
    return split_indices[mask]


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

def highlight(img_like, places, axis=0):
    img = np.zeros_like(img_like)
    if axis==0:
        img[places, :] = 1
    else:
        img[:, places] = 1
    return img

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
            print("is_hole: " + str(is_hole))
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

            if (is_hole):
                x = None
                # debug_draw(word, None, red_word)
            else:
                # debug_draw(word, box_mask, red_word)
                # print(max_width)
                subwords.append([ max_width, word[:, min_width:max_width+1], filled_word[:, min_width:max_width+1] ])
            print("\n")
            cnt += 1
        
        subwords.sort(key=sorting_key, reverse=True)
        word_characters = []

        for [_, subword_pure, subword_filled] in subwords:
            subword_characters = _segment_subword(subword_pure, subword_filled)
            word_characters += subword_characters

        words_characters.append(word_characters)
    return words_characters