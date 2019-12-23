
import cv2 as cv
from common import *
from preprocess import *

def get_split_index(seg_cntr, max_seg):
    if seg_cntr < 0:
        return 0
    elif seg_cntr == max_seg-1:
        return max_seg-2
    else:
        return seg_cntr


# check if the segment has a hole
def has_hole(word):
    _, contours, heirarchy = cv.findContours(word.astype('uint8')*255, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if (len(contours) == 0):
        return False
    is_hole = heirarchy [:,:,3] != -1
    is_hole = is_hole[0]
    return np.max(is_hole) > 0


def fix_SAD(curr_segment, next_segment, seg_cntr, n_seg, mask):
    curr_sx, curr_sy = curr_segment.shape
    next_sx, next_sy = next_segment.shape
    if has_hole(curr_segment) and not has_hole(next_segment):
        # Candidate to be a SAD/DAD
        if (curr_sy > 2 * next_sy and curr_sx > next_sx and next_sx < next_sy - 1):
            mask[get_split_index(seg_cntr, n_seg)] = False
            seg_cntr += 2
            print("Over-segmented SAD detected.")
            return True, seg_cntr, mask
            # debug_draw(curr_segment)
            # debug_draw(next_segment)
    return False, seg_cntr, mask

def fix_NUN(curr_segment, next_segment, seg_cntr, n_seg, mask, segments):
    curr_sx, curr_sy = curr_segment.shape
    next_sx, next_sy = next_segment.shape
    if next_sx > 2 * next_sy and curr_sx >= curr_sy and not(curr_sx >= 2 * curr_sy):
        # Candidate to be an over-segmented noon.
        horizontal_projection = np.sum(segments[seg_cntr], axis=1)
        baseline = np.argmax( horizontal_projection )
        print("baseline: ", baseline)
        if (baseline >= 18 and baseline < 20):
            print("Over-segmented NUN detected.")
            mask[get_split_index(seg_cntr, n_seg)] = False
            seg_cntr += 2
            return True, seg_cntr, mask
    return False, seg_cntr, mask


def fix_SEEN(curr_segment, next_segment, next2_segment, seg_cntr, n_seg, mask, segments):
    curr_sx, curr_sy = curr_segment.shape
    next_sx, next_sy = next_segment.shape
    next2_sx, next2_sy = next2_segment.shape
    print("next2 info: ", next2_sx, next2_sy)
    # Candidate to be a SEEN/SHEEN
    rcurrent = curr_sx / curr_sy
    rnext = next_sx / next_sy
    rnext2 = next2_sx / next2_sy
    rcurnext = rcurrent / rnext
    rnext2next  = rnext2 / rnext
    print("ratios: ", rcurrent / rnext, rnext2 / rnext )
    hp1 = np.sum(segments[seg_cntr], axis=1)
    bl1 = np.argmax(hp1)
    hp2 = np.sum(segments[seg_cntr+1], axis=1)
    bl2 = np.argmax(hp2)
    hp3 = np.sum(segments[seg_cntr+2], axis=1)
    bl3 = np.argmax(hp3)
    print("Baselines: ", bl1, bl2, bl3)
    print(np.mean(hp1), bl1, np.mean(hp2), bl2, np.mean(hp3), bl3)

    if has_hole(curr_segment) or has_hole(next_segment):
        return False, seg_cntr, mask

    if (np.abs(bl1 - bl2) < BASELINE_THICKNESS and bl3 > bl2 + 2 and bl3 > bl1 + 2 and np.abs(rnext2next - 1.03) < 0.3 ):
        # Baselines: 20 20 26 - 18 18 25
        # ratios: 1.607 1.03 - 2.57 1.2375
        # NOT: 
        print("Over-segmented END SEEN detected.")
        mask[get_split_index(seg_cntr, n_seg)] = False
        mask[get_split_index(seg_cntr+1, n_seg)] = False
        seg_cntr += 3
        return True, seg_cntr, mask
    if (np.abs(bl1 - bl2) <= BASELINE_THICKNESS and np.abs(bl2 - bl3) <= BASELINE_THICKNESS and np.abs(bl1 - bl3) <= BASELINE_THICKNESS and rcurnext > 1 and rcurnext < 2.2 and rnext2next < 1.1 and rnext2next > 0.65):
        print("Over-segmented MIDDLE SEEN detected.")
        # 2.0625 1.05
        # 1.3333 0.909
        # 1.875 0.8333
        # 1.6 1.03
        # 1.16 0.687                    
        mask[get_split_index(seg_cntr, n_seg)] = False
        mask[get_split_index(seg_cntr+1, n_seg)] = False
        seg_cntr += 3
        return True, seg_cntr, mask
    return False, seg_cntr, mask


def fix_horizontal_stroke(curr_segment, seg_cntr, n_seg, mask, segments):
    local_baseline_thickness = 2
    curr_sx, curr_sy = curr_segment.shape
    horizontal_projection = np.sum(segments[seg_cntr], axis=1)
    baseline = np.argmax( horizontal_projection )
    baseline_cropped_idx = np.argmax( np.sum(curr_segment, axis=1) )
    baseline_val = np.sum(horizontal_projection[baseline-1:baseline+2])
    above_bl_val = np.sum(horizontal_projection[:baseline-1])
    
    highlighted = highlight(curr_segment, baseline_cropped_idx)
    # debug_draw(curr_segment, highlighted)
    
    if curr_sx <=1 or (not has_hole(curr_segment) and curr_sx <= 4 and curr_sy >= 3 * curr_sx and baseline_cropped_idx > 0):
        print("Probably a stroke..")
        mask[get_split_index(seg_cntr-1, n_seg)] = False
        seg_cntr += 1
        return True, seg_cntr, mask
    return False, seg_cntr, mask





# TODO: MANY FIXES.
def fix_segments(segments, split_indices):
    seg_cntr = 0
    n_deleted = 0
    n_seg = len(segments)
    mask = np.ones_like(split_indices, dtype=bool)
    while seg_cntr < n_seg:
        curr_segment = image_autocrop(segments[seg_cntr])
        curr_sx, curr_sy = curr_segment.shape
        print("Current info for segment " + str(seg_cntr) + ": ", curr_sx, curr_sy)
        # debug_draw(segments[seg_cntr])
        fixed_hstroke, seg_cntr, mask = fix_horizontal_stroke(curr_segment, seg_cntr, n_seg, mask, segments)
        if fixed_hstroke:
            continue
        if seg_cntr < (n_seg - 1):
            next_segment = image_autocrop(segments[seg_cntr + 1])
            next_sx, next_sy = next_segment.shape
            print("Next info: ", next_sx, next_sy)
            fixed_sad, seg_cntr, mask = fix_SAD(curr_segment, next_segment, seg_cntr, n_seg, mask)
            if fixed_sad:
                continue
            fixed_nun, seg_cntr, mask = fix_NUN(curr_segment, next_segment, seg_cntr, n_seg, mask, segments)
            if fixed_nun:
                continue
            if seg_cntr < (n_seg - 2):
                next2_segment = image_autocrop(segments[seg_cntr + 2])
                fixed_SEEN, seg_cntr, mask = fix_SEEN(curr_segment, next_segment, next2_segment, seg_cntr, n_seg, mask, segments)
                if fixed_SEEN:
                    continue
        print("\n")
        seg_cntr += 1
    print("fix_segments: END.")
    return split_indices[mask]
