
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


# TODO: FIX ALL THE SPLITS.
    # RULES:
    # 1. SAD/DAD:
    #    If I have a hole, the next has low width, and am much higher than the next.
def fix_segments(segments, split_indices):
    seg_cntr = 0
    n_seg = len(segments)
    for seg_cntr in range(n_seg):
        curr_segment = segments[seg_cntr]
        if seg_cntr < (n_seg - 1):
            next_segment = segments[seg_cntr + 1]
            if has_hole(curr_segment):
                debug_draw(curr_segment)


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
    split_segments = []
    start = width-1
    for split_index in split_indices:
        segment = word_skeleton[:, split_index:start+1]
        start = split_index
        split_segments.append(segment)
    split_segments.append(word_skeleton[:, :start+1])
    fix_segments(split_segments, split_indices)

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
        print("Probably isolated..")
        debug_draw(raw_cropped, raw_cropped)
        return [ raw, filled ]

    # 1. find the baseline in the filled image
    horizontal_projection = np.sum(filled, axis=1)
    baseline = np.argwhere(horizontal_projection == np.max(horizontal_projection))
    baseline_highlight = highlight(filled, baseline, 0)
    # debug_draw(raw, baseline_highlight)

    detect_mid_chars(raw, filled)




def segment_characters_habd(words):
    words_characters = []
    for word in words:
        show_images([word])
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
                subwords.append([ word[:, min_width:max_width+1], filled_word[:, min_width:max_width+1] ])
            print("\n")
            cnt += 1
        
        for [subword_pure, subword_filled] in subwords:
            characters = _segment_subword(subword_pure, subword_filled)

        # thinned_word = thin(word)
        # baseline = np.argmax( np.sum(word, axis=1) ) + 3
        # baseline_top = baseline - 4
        # new_word = np.zeros_like(thinned_word)
        # new_word[baseline_top:baseline+1, :] = 1
        # debug_draw(word, blue=new_word)
        # detect_mid_chars(thinned_word)
        # debug_draw(thinned_word)

        # segp_word = np.zeros_like(thinned_word)
        # for w in range(width):
        #     val_above_baseline = np.sum(thinned_word[:baseline-2, w])
        #     val_baseline = np.average( [word[baseline, w], word[baseline-1, w], word[baseline+1,w]])
        #     val_below_baseline = np.sum(thinned_word[baseline+2:, w])
        #     print(val_above_baseline, val_baseline, val_below_baseline)
        #     if val_baseline > 0 and (val_above_baseline == 0 and val_below_baseline == 0):
        #         segp_word[baseline, w] = 1
        #         print("Here!")
        # debug_draw(thinned_word, segp_word)

        # show_images([word_skeleton])
    return word_characters