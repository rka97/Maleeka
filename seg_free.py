import numpy as np
from util import *
from skimage import io
from segmentation import *
from scipy.cluster.vq import kmeans
from sklearn.cluster import MiniBatchKMeans, KMeans
import pickle
import os

# Credits to https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi/32681075#32681075
def encode_column(column, num_transitions=7):
    col_arr = np.asarray(column)
    n = len(col_arr)
    runs_vector = np.zeros(num_transitions)
    if n == 0:
        return runs_vector
    else:
        transitions = np.array(col_arr[1:] != col_arr[:-1])
        aug_transitions = np.append(np.where(transitions), n - 1)
        runs = np.diff(np.append(-1, aug_transitions))
        if (col_arr[0] == 0):
            runs_zero_start = runs
        else:
            runs_zero_start = np.zeros(runs.size + 1)
            runs_zero_start[0] = 0
            runs_zero_start[1:] = runs
        num_runs_to_take = np.min([len(runs_zero_start), num_transitions])
        runs_vector[:num_runs_to_take] = runs_zero_start[:num_runs_to_take]
        return runs_vector


def khorshed_rle(lines, dataset):
    for line in lines:
        height, width = line.shape
        w = width - 1
        while w >= 0:
            dataset.append(encode_column(line[:, w]))
            w -= 1
    print("Number of feature vectors extracted from file is: " + str(len(dataset)))
    return dataset


def train_codebook(file_names):
    kmeans = MiniBatchKMeans(NUM_CLUSTERS)
    try:
        with open(CB_CODEBOOK_PATH, "rb") as file_handle:
            kmeans = pickle.load(file_handle)
            return kmeans
    except:
        print("Codebook doesn't exist, creating it...")
    for file_name in file_names:
        print("Processing " + file_name + "...")
        image = io.imread(file_name)
        image = preprocess(image)
        image_lines = segment_lines(image)
        file_features = khorshed_rle(image_lines, [])
        print(file_features)
        kmeans.partial_fit(file_features)
    with open(CB_CODEBOOK_PATH, "ab+") as file_handle:
        pickle.dump(kmeans, file_handle)
    return kmeans


def get_features_from_image(file_name, character_mode=True):
    image = io.imread(file_name)
    image = preprocess(image, not(character_mode))
    image_lines = segment_lines(image, character_mode)
    file_features = khorshed_rle(image_lines, [])
    return file_features


def load_character_data_samples(kmeans: KMeans):
    try:
        with open(LETTERS_PATH, "rb") as file_handle:
            letters = pickle.load(file_handle)
            return letters
    except:
        print("Letter samples don't exist, creating them...")
    letters = init_letters.copy()
    for letter_name, letter_data in letters.items():
        print("Processing " + letter_name + " " + letter_data["Letter"])
        forms = letter_data["Forms"]
        for form in forms:
            j = 0
            character_dir = CHARACTERS_PATH + letter_name + "/" + form
            files = os.listdir(character_dir)
            character_data = []
            for file in files:
                file_name = character_dir + "/" + file
                if j == 0:
                    j += 1
                file_features = np.array(get_features_from_image(file_name))
                predictions = kmeans.predict(file_features)
                character_data.append(predictions)
            letters[letter_name][form] = character_data
    with open(LETTERS_PATH, "ab+") as file_handle:
        pickle.dump(letters, file_handle)
    return letters

    