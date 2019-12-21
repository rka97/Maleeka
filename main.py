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
from seg_free import *
from segmentation import *
from hmms import *

THRESH = 0.001
LINE_HEIGHT = 60

def main():
    # files = ["images/capr2.png", "images/capr3.png", "images/capr4.png"]
    # create_cb_dataset(files)
    file_names = []
    for i in np.arange(start=1, stop=11, step=1):
        file_names.append("images/scanned/csep" + str(i) + ".png")
    kmeans = train_codebook(file_names)
    # print(kmeans.cluster_centers_)
    letters_data = load_character_data_samples(kmeans)
    character_models = train_character_models(letters_data)
    big_model = get_big_model(character_models)


    # train_character_models(kmeans)
    # train_character_model(["images/characters/1575/End/NotoSansArabicUI-ExtraCondensedBlack.png"])
    # print(train_character_models())
    # train_character_models()
    # image = io.imread("images/scanned/capr2.png")
    # image = preprocess(image)
    # image_lines = segment_lines(image)
    # for line in image_lines:
    #     show_image([line])


main()