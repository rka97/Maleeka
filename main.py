import deskew
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import skeletonize, thin, binary_opening
from skimage.transform import rescale, resize, rotate
from skimage.graph import route_through_array
import time
import cv2 as cv

THRESH = 0.001
DIRECTORY = '../'
def main():
    start_time = time.time()
    img_pos = DIRECTORY +'capr61.png'
    image = io.imread(img_pos)
    image = preprocess(image)
    io.imsave('output.png', image * 255)
    image_lines = segment_lines(image)
    image_words = segment_words(image_lines)
    ## Now need to segment words into characters.
    #image_characters = segment_characters_latifa(image_words)
    segment_characters_qaroush(image_words)
    end_time = time.time()
    print("Elapsed Time = "+ str(end_time - start_time) + "secs")


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
    #show_image([thresholded_image])
    return thresholded_image.astype(int)


# The goal of this is to segment the image into lines.
def segment_lines(img):
    horizontal_projection = np.sum(img, axis=1) / 255
    # horizontal_projection = gaussian(horizontal_projection, sigma=0.5) # smooth everything a little..
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
            #io.imsave(DIRECTORY+'images/lines/line' + str(len(lines)) +'.png', resized_line * 255)
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
        vertical_projection = np.reshape(vertical_projection, (vertical_projection.shape[0], 1))
        vertical_projection = gaussian(vertical_projection, sigma=1.0)
        threshold = np.max(vertical_projection) * 0.02
        # vertical_projection = gaussian(vertical_projection, sigma=1.5)	
        # threshold = np.max(vertical_projection) * 0.1
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


# this should be done on lines, try on words to see if there is a difference
# try a more optimized implementation
def get_max_transitions(words):
    max_transitions = []
    for word in words:
        # line = thin(line)
        # line = skeletonize(line)
        word_no_dots = word#fill_dots(word)
        max_transition = 0
        max_transition_index = 0
        horizontal_projection = np.sum(word, axis=1)
        baseline = np.argmax(horizontal_projection)
        i = baseline
        while i >= 0: #< line.shape[0]: # not sure of this
            current_transition = 0
            flag = 0
            j = word_no_dots.shape[1] - 1
            while j >= 0:
                if word_no_dots[i,j] == 0 and flag == 1:
                    current_transition += 1
                    flag = 0
                elif word_no_dots[i,j] != 0 and flag == 0:
                    flag = 1
                j -= 1
            if current_transition >= max_transition:
                max_transition = current_transition
                max_transition_index = i
            i -= 1
        max_transitions.append(max_transition_index)
    return max_transitions

# function to get the array index closest to the key whose element = 0
def find_nearest_zero(arr, key):
    for k in range(len(arr)):
        if arr[key-k] ==  0:
            return key - k
        elif arr[key+k] == 0:
            return key + k
    return key

# function that fills the dots in the image, needed for getting max transitions and stroke segments
def fill_dots(word_with_dots):
    word = word_with_dots.copy()
    _, contours, heirarchy = cv.findContours(word.copy().astype('uint8')*255, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
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

# find index with the nearest number to the key
def find_nearest_index(arr, key):
    index = next((x for x, val in enumerate(arr) if val >= key), None)
    if index == None:
        return len(arr) - 1
    if (arr[index] - key > arr[index-1] - key):
        index = index - 1
    return arr[index]

# get candidate cut points
def get_cut_points(words, max_transitions):
    separation_regions = []
    word_index = np.zeros(len(words))
    index = 0
    for word in words:
        #line= thin(line)
        #word= binary_opening(word)
        vp = np.sum(word, axis=0)
        # try opening (erosion) here if results are unsatisfactory  ==> tried it.. baaad results, try opening
        # without skeletonizing next
        mfv = mode(vp)[0][0]
        flag = 0
        cut_index = -1
        start_index = -1
        mid_index = -1
        end_index = -1
        for i in range(word.shape[1]):
            if word[max_transitions[index],i] == 0 and flag == 0:
                end_index = i
                flag = 1
            elif word[max_transitions[index],i] != 0 and flag == 1:
                start_index = i
                mid_index = round((start_index + end_index) / 2)
                flag = 0
                # if (start_index != -1 and end_index != -1):
                    # s = line.copy()
                    # s[0:5, end_index: start_index] = 1
                    #show_image([s[:,end_index:start_index]])
                # unneeded, this finds words
                if np.count_nonzero(vp[end_index:start_index] == 0) != 0:
                   #find closest zero to the mid_index
                    cut_index = find_nearest_zero(vp, mid_index)
                elif vp[mid_index] == mfv:
                    cut_index = mid_index
                else:
                    smaller_arr = []
                    mid_arr = []
                    for k in range (end_index, start_index+1):
                        if vp[k] <= mfv and k <= mid_index and k >= end_index:
                            smaller_arr.append(k)
                        if vp[k] <= mfv and k >=mid_index and k <= start_index:
                            mid_arr.append(k)
                    if (len(smaller_arr) != 0):
                        cut_index = find_nearest_index(smaller_arr, mid_index)
                    elif (len(mid_arr) != 0):
                        cut_index = find_nearest_index(mid_arr, mid_index)
                    else:
                        cut_index = mid_index
            if (start_index != -1 and mid_index != -1 and end_index != -1 and cut_index != -1):
                separation_regions.append([int(cut_index), int(start_index), int(mid_index), int(end_index)])
                cut_index = start_index = mid_index = end_index = -1
        word_index[index] = len(separation_regions)     #to know the number of separation regions a word has
        index += 1
    return separation_regions, word_index

# check to see if the start and end are connected on the baseline
def is_baseline(word, start, end, baseline):
    for i in range(end, start+1):
        if (word[baseline, i] == 0):
            return False
    return True

# check if the segment has a hole
def has_hole(word, sr1, sr2, cut_idx):
    _, contours, heirarchy = cv.findContours(word[:,sr1:sr2].copy().astype('uint8')*255, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if (len(contours) == 0):
        return False
    is_hole = heirarchy [:,:,3] != -1
    is_hole = is_hole[0]
    i = 0 
    while i < len(contours):
        if (is_hole[i]):
            if cut_idx - sr1 in contours[i][:,:,0][:,0]:
                return True
        i += 1
    return False

#see if there is a numpy replacement
#gets height (no of connected white pxels)
def get_height_no_dots(segment, baseline):
    start = baseline
    end = baseline
    i =  baseline
    while i < (segment.shape[0] - 1):
        if (np.count_nonzero(segment[i,:] == 1) != 0):
            end = i
            break
        i += 1
        if i == segment.shape[0] - 1:
            end = i
    
    i = baseline
    while i  > 0:
        if (np.count_nonzero(segment[i,:] == 1) != 0):
            start = i
            break
        i -= 1
        if i == 0:
            start = i
    return start, end

# prep for stroke checking 
def get_stroke(segment):
    start = 0
    end = 0
    flag = 0
    for i in range(segment.shape[0]):
        if np.count_nonzero(segment[i,:] == 1) != 0:
            if (flag == 0):
                start = i
                flag = 1
            else:
                end = i
    return segment[start:end,:]

# check if thesegment is a stroke, still haven't figured out what stroke means
def is_stroke(word_no_dots, sr, baseline, mfv, prev_sr):
    '''
    segment is a stroke if it is:
    (i) single connected component,
    (ii) the sum of horizontal projection above baseline is greater than the sum of horizontal projection below baseline,
    (iii) the height of the segment is less than twice the second peak value of the horizontal projection,
    (iv) the mode value of the horizontal projection is equal to MFV value, and
    (v) the segment has no Holes. 
    '''
    segment = get_stroke(word_no_dots[:,sr[0]:prev_sr[0]].copy())
    if (segment.shape[0] == 0):
        return False
    above_baseline_hp = np.sum(np.sum(segment[0:baseline,:], axis=1))
    below_baseline_hp = np.sum(np.sum(segment[baseline:-1,:], axis=1))
    horizontal_projection = np.sum(segment, axis=1)
    hpmode = mode(horizontal_projection)
    horizontal_mode = hpmode[0][0]
    segment_height = get_height(segment)
    no_of_componenets, _ = cv.connectedComponents(np.uint8(segment),connectivity=8)
    sort_hp = sorted(horizontal_projection)
    second_peak = sort_hp[-2]
    #print("sr = ",sr)
    seg_has_hole = has_hole(word_no_dots, sr[3], sr[1], sr[0])
    if (above_baseline_hp > below_baseline_hp and horizontal_mode == mfv and seg_has_hole == False and segment_height < 2 * second_peak and no_of_componenets == 2):
        return True
    return False

# find if the segment has dots
def has_dots(segment, baseline):
    above_baseline = segment[0:baseline,:]
    below_baseline = segment[baseline:-1,:]
    componenets_above_baseline, _ = cv.connectedComponents(np.uint8(above_baseline),connectivity=8)
    componenets_below_baseline, _ = cv.connectedComponents(np.uint8(below_baseline),connectivity=8)
    if (componenets_above_baseline > 2 or componenets_below_baseline > 2):
        return True
    return False

# get the height of a segment
def get_height(segment):
    start = 0
    end = 0
    flag = 0
    for i in range(segment.shape[0]):
        if np.count_nonzero(segment[i,:] == 1) != 0:
            if (flag == 0):
                start = i
                flag = 1
            else:
                end = i
    return end - start

# find of a path exists between 2 pixels
def has_path(img, start, end):
    search_array = (img == 0) * 1.0
    base_value = search_array[start[0], start[1]] + search_array[end[0], end[1]]
    #_, cost = route_through_array( search_array, start=(start[0], start[1]), end=(end[0], end[1]-1), fully_connected=True, geometric= False)
    _, cost = route_through_array( search_array, start=(start[0], start[1]), end=(end[0], end[1]), fully_connected=True, geometric= False)
    return cost == base_value

# check if a segment is empty	
def empty_segment(segment):	
    pixels_count = np.count_nonzero(segment == 0)	
    print("count = ", pixels_count)	
    if (pixels_count <= 108):	
        return True	
    return False	

# filtration of candidate cut indices
def separation_region_filteration(words, separation_regions, word_index, max_transitions):
    i = 0
    prev_index = 0
    valid_separation_regions = []
    separation_indices = np.zeros(len(words))
    for _, word in enumerate(words):
        # if idx == 106:
        #     print("hey stop")
        #     pass
        vp = np.sum(word, axis=0)
        mfv = mode(vp)[0][0]
        horizontal_projection = np.sum(word, axis=1)
        baseline = np.argmax(horizontal_projection)
        line_height = get_height(word)
        word_no_dots = fill_dots(word.copy())
        srs = separation_regions[prev_index:int(word_index[i])]
        prev_sr = [word.shape[1] - 1, word.shape[1] - 1, word.shape[1] - 1, word.shape[1] - 1]
        j = len(srs)
        while j > 0:
            j -= 1
            #show_image([word])
            segment = word[:,srs[j][0]:prev_sr[0]]
            #show_image([segment])
            if vp[srs[j][0]] == 0:
                valid_separation_regions.append(srs[j])
                prev_sr = srs[j]
                continue
            if  not has_path(word, (max_transitions[i],srs[j][1]), (max_transitions[i],srs[j][3])):    #Should I start from the baseline or the transition point?
                valid_separation_regions.append(srs[j])
                prev_sr = srs[j]
                continue
            if j-1 >= 0:
                if(has_hole(word,srs[j-1][0],prev_sr[0], srs[j][0])):
                    prev_sr = srs[j]
                    continue
            if not has_path(word, (baseline, srs[j][1]), (baseline, srs[j][3])):
            #if not is_baseline(word, srs[j][1], srs[j][3], baseline):
                above_baseline_hp = np.sum(np.sum(word[0:baseline,srs[j][3]:srs[j][1]], axis=1))
                below_baseline_hp = np.sum(np.sum(word[baseline:-1,srs[j][3]:srs[j][1]], axis=1))
                if below_baseline_hp > above_baseline_hp:
                    prev_sr = srs[j]
                    continue
                elif vp[srs[j][0]] < mfv:
                    valid_separation_regions.append(srs[j])
                    prev_sr = srs[j]
                    continue
                else:  # why do this ? :\\
                    prev_sr = srs[j]
                    continue
            if j == 0:
                prev_sr = srs[j]
                continue
            if(j - 1 >= 0):
                if vp[srs[j-1][0]] == 0 :
                    if (get_height(word[:,srs[j][0]:prev_sr[0]]) < (line_height) / 2):
                        prev_sr = srs[j]
                        continue
            if (is_stroke(word_no_dots, srs[j], baseline, mfv, prev_sr) == False):
                if (j-1 >= 0):
                    if not has_path(word, (baseline, srs[j-1][1]), (baseline, srs[j-1][3])) and vp[srs[j-1][0]] <= mfv:
                    #if not is_baseline(word, srs[j-1][1], srs[j-1][3], baseline) and vp[srs[j-1][0]] <= mfv:
                        prev_sr = srs[j]
                        continue
                    else:
                        valid_separation_regions.append(srs[j])
                        prev_sr = srs[j]
                        continue

            if(is_stroke(word_no_dots, srs[j], baseline, mfv, prev_sr) == True and has_dots(segment, baseline) == True):
                valid_separation_regions.append(srs[j]) 
                prev_sr = srs[j]
                continue
            if j-1 >= 0:
                if is_stroke(word_no_dots, srs[j], baseline, mfv, prev_sr) == True and has_dots(segment, baseline) == False:
                    if j-2 >= 0:
                        if (is_stroke(word_no_dots, srs[j-1] , baseline, mfv, srs[j-2]) == True) and has_dots(word[:,srs[j-2][0]:srs[j-1][0]], baseline) == False:
                            valid_separation_regions.append(srs[j])
                            j -= 2
                            if j >= 0:
                                prev_sr = srs[j]
                            continue
                    if j-3 >= 0:
                        if (is_stroke(word_no_dots, srs[j-1] , baseline, mfv, srs[j-2]) == True) and has_dots(word[:,srs[j-2][0]:srs[j-1][0]], baseline) == True and (is_stroke(word_no_dots, srs[j-2] , baseline, mfv, srs[j-3]) == True) and has_dots(word[:,srs[j-3][0]:srs[j-2][0]], baseline) == False:
                            valid_separation_regions.append(srs[j])
                            j -= 2
                            if j >= 0:
                                prev_sr = srs[j]
                            continue
                    if j-2 >= 0:
                        if (is_stroke(word_no_dots, srs[j-1] , baseline, mfv, srs[j-2]) == False) or ((is_stroke(word_no_dots, srs[j-1] , baseline, mfv, srs[j-2]) == True) and has_dots(word[:,srs[j-2][0]:srs[j-1][0]], baseline) == True):
                            prev_sr = srs[j]
                            continue
        separation_indices[i] = len(valid_separation_regions)
        i += 1
        prev_index = int(word_index[i-1])
    return valid_separation_regions, separation_indices

# show image of max transitions
def show_max_transitions(words, max_transitions):
    i = 0
    for word in words:
        word[max_transitions[i],:] = 1
        show_image([word])
        i += 1

# show images of cut points
def show_cut_points(words, separation_regions, word_index, max_transitions):
    i = 0
    prev_index = 0
    for word in words:
        srs = separation_regions[prev_index:int(word_index[i])]
        word[max_transitions[i],:] = 1
        for sr in srs:
            word[:,sr[2]] = 1
            # line[0:5, sr[1]] = 1
            # line[5:10, sr[2]] = 1
            # line[10:15, sr[0]] = 1
            # line[15:20, sr[3]] = 1
            print("cut = "+ str(sr[0])+ "   start = "+ str(sr[1])+"   mid = "+ str(sr[2])+"    end = "+ str(sr[3]))
        i += 1
        show_image([word])
        prev_index = int(word_index[i-1])

# show images after filteration
def show_segments(words, valid_separation_regions, separation_indices):
    i = 0
    prev_index = 0
    for word in words:
        srs = valid_separation_regions[prev_index:int(separation_indices[i])]
        prev_sr = word.shape[1] - 1
        w = word.copy()
        for sr in srs:
            w[0:5, sr[1]] = 1
            w[5:10, sr[2]] = 1
            w[10:15, sr[0]] = 1
            w[15:20, sr[3]] = 1
            print("cut = "+ str(sr[0])+ "   start = "+ str(sr[1])+"   mid = "+ str(sr[2])+"    end = "+ str(sr[3]))
            segment = word[:,sr[0]: prev_sr]
            if (not empty_segment(segment)):
                show_image([segment])
            prev_sr = sr[0]
        if (len(srs) == 0):
            segment = word
        else:
            segment = word[:,:srs[-1][0]]
        if (not empty_segment(segment)):
             show_image([segment])
        show_image([w])
        i += 1
        prev_index = int(separation_indices[i-1])

def segment_characters_qaroush(words):
    '''
    1. baseline detection
    2. find max_transitions
    3. cut point identification
    4. filteration
    5. character extraction
    '''
    # 1 and 2
    max_transitions = get_max_transitions(words)
    # show_max_transitions(words, max_transitions)

    # 3
    separation_regions, word_index = get_cut_points(words, max_transitions)
    # show_cut_points(words, separation_regions, word_index, max_transitions)

    #4
    valid_separation_regions, separation_indices = separation_region_filteration(words, separation_regions, word_index, max_transitions)
    show_segments(words, valid_separation_regions, separation_indices)
    

main()
