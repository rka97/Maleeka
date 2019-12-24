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
# from segment_char import *
from generate_dataset import *
from word_based import *
from os import path

DATASET_DIRECTORY = 'images/'

def get_raw_dataset_files(months, num, start=1):
    image_files = []
    text_files = []
    for month in months:
        for i in range(num):
            img_name = month + str(i+start) + '.png'
            image_files.append( DATASET_DIRECTORY + 'scanned/' + img_name  )
            text_files.append(  DATASET_DIRECTORY + 'text/' + img_name.split('.')[0] + '.txt'  )
    return image_files, text_files


def main():
    train_months = ['csep', 'capr', 'cjan', 'cjun']
    image_files, text_files = get_raw_dataset_files(train_months, 400)
    if not (path.exists(LEXICON_PATH) and path.exists(KMEANS_PATH) and path.exists(FEATURES_MAT_PATH)):
        lexicon, kmeans, features_mat_arr, cluster_idx = wb_generate_dataset(image_files, text_files)

    print("Loading dataset..")
    lexicon = load_must_exist(LEXICON_PATH)
    kmeans = load_must_exist(KMEANS_PATH)
    features_mat_arr = np.array(load_must_exist(FEATURES_MAT_PATH))
    labels = kmeans.predict(features_mat_arr)
    cluster_idx = {i: np.where(labels == i)[0] for i in range(kmeans.n_clusters)}
    print("Dataset loaded!")
    
    test_months = ['cjan', 'cfeb', 'cmar', 'capr', 'cmay', 'cjun', 'cjul', 'caug', 'csep', 'coct', 'cnov', 'cdec']
    test_image_files, test_text_files = get_raw_dataset_files(test_months, 5, start=705)
    wb_run_test(test_image_files, test_text_files, lexicon, kmeans, features_mat_arr, cluster_idx)

main()
