from skimage import io
import pathlib
from common import *
from preprocess import *
from scipy.fftpack import dct
from skimage.measure import regionprops
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import cKDTree
import pickle
import ntpath
import shutil 
import timeit

N_ZIGZAG = 100
SQRT_N_ZIGZAG = int(np.math.sqrt(N_ZIGZAG))
N4_ZIGZAG = 25
SQRT_N4_ZIGZAG = int(np.math.sqrt(N4_ZIGZAG))
CODEBOOK_SIZE = 1024
LEXICON_PATH = "dataset/lexicon.data"
KMEANS_PATH = "dataset/kmeans.data"
FEATURES_MAT_PATH = "dataset/features.data"
PROCESSED_FILES_PATH = "dataset/processed.data"

def compare(xy):
    x, y = xy
    return (x + y, -y if (x + y) % 2 else y)

def zigzag(n):
    '''zigzag rows'''
    xs = range(n)
    return {index: n for n, index in enumerate(sorted(
        ((x, y) for x in xs for y in xs),
        key=compare
    ))}

zigzag_per_n = {}

def get_zigzag_keys(dim_size):
    if dim_size in zigzag_per_n:
        return zigzag_per_n[dim_size]
    else:
        zigzag_keys_rows = []
        zigzag_keys_columns = []
        zigzag_for_n = zigzag(dim_size)
        for key, value in zigzag_for_n.items():
            zigzag_keys_rows.append(key[0])
            zigzag_keys_columns.append(key[1])
        zigzag_keys = [zigzag_keys_rows, zigzag_keys_columns]
        zigzag_per_n[dim_size] = zigzag_keys
        return zigzag_keys

def get_dct_features(n_sqrt, image):
    zigzag_keys = get_zigzag_keys(n_sqrt)
    dct_result = dct(dct(image.T).T)
    features_vec = dct_result[tuple(zigzag_keys)]
    return features_vec

def word_features_vec(image):
    zigzag_keys = get_zigzag_keys(SQRT_N_ZIGZAG)

    N_sqrt = np.min([SQRT_N_ZIGZAG, image.shape[0], image.shape[1]])
    if N_sqrt == SQRT_N_ZIGZAG:
        dct_result = dct(dct(image.T).T)
        features_list = [dct_result[tuple(zigzag_keys)]]
    else:
        small_features = get_dct_features(N_sqrt, image)
        long_features = np.zeros(N_ZIGZAG)
        long_features[:small_features.size] = small_features
        features_list = [ long_features ]

    properties = regionprops(image, image)
    center_of_mass = properties[0].centroid
    sx, sy = np.array(center_of_mass, dtype=int)

    subimages = [ image[:sx,:sy], image[sx:,:sy], image[:sx,sy:], image[sx:,sy:] ]
    for subimage in subimages:
        n_sqrt = np.min([SQRT_N4_ZIGZAG, subimage.shape[0], subimage.shape[1]])
        features_subimage = get_dct_features(n_sqrt, subimage)
        feat_subimage = np.zeros(N4_ZIGZAG)
        feat_subimage[:features_subimage.size] = features_subimage
        features_list.append(feat_subimage)
    features_vec = np.concatenate(features_list)

    return features_vec

# TODO: remove this, use word_features_vec and list concatenation instead.
def get_word_features(word_images, word_texts, lexicon, kmeans: MiniBatchKMeans, features_mat):
    zigzag_keys = get_zigzag_keys(SQRT_N_ZIGZAG)

    image_features = []
    for i in range(len(word_images)):
        image = word_images[i]
        text = word_texts[i]
        N_sqrt = np.min([SQRT_N_ZIGZAG, image.shape[0], image.shape[1]])
        if N_sqrt == SQRT_N_ZIGZAG:
            dct_result = dct(dct(image.T).T)
            features_list = [dct_result[tuple(zigzag_keys)]]
        else:
            small_features = get_dct_features(N_sqrt, image)
            long_features = np.zeros(N_ZIGZAG)
            long_features[:small_features.size] = small_features
            features_list = [ long_features ]

        properties = regionprops(image, image)
        center_of_mass = properties[0].centroid
        sx, sy = np.array(center_of_mass, dtype=int)

        subimages = [ image[:sx,:sy], image[sx:,:sy], image[:sx,sy:], image[sx:,sy:] ]
        for subimage in subimages:
            n_sqrt = np.min([SQRT_N4_ZIGZAG, subimage.shape[0], subimage.shape[1]])
            features_subimage = get_dct_features(n_sqrt, subimage)
            feat_subimage = np.zeros(N4_ZIGZAG)
            feat_subimage[:features_subimage.size] = features_subimage
            features_list.append(feat_subimage)
        features_vec = np.concatenate(features_list)
        image_features.append(features_vec)
        lexicon.append(text)
    
    features_mat += image_features
    return features_mat, lexicon, kmeans


def recognize_word(word_image, kmeans: MiniBatchKMeans, cluster_idx, features_mat):
    word_features = word_features_vec(word_image)
    prediction_cluster = (kmeans.predict( np.reshape(word_features, (1, -1))))[0]
    prediction = kmeans.cluster_centers_[prediction_cluster]
    words_in_cluster = cluster_idx[prediction_cluster]
    feat_submatrix = features_mat[words_in_cluster]
    indices_submatrix = (np.arange(len(features_mat)))[words_in_cluster]
    tree = cKDTree(np.array(feat_submatrix))
    nearest_idx = tree.query(word_features, k=1)[1]
    return indices_submatrix[nearest_idx]


def recognize_image(image_file_name, features_mat, lexicon, kmeans: MiniBatchKMeans, cluster_idx):
    print("Recognizing %s" % (image_file_name))
    image = io.imread(image_file_name)

    start = timeit.default_timer()
    
    image = preprocess(image)
    image_lines = segment_lines(image)
    image_words = segment_words(image_lines)
    i = 0
    output = ""
    for word in image_words:
        word_idx = recognize_word(word, kmeans, cluster_idx, features_mat)
        output = output + lexicon[word_idx] + " "
    
    stop = timeit.default_timer()
    return output, stop - start


def wb_run_test(image_files, truth_text_files, lexicon, kmeans, features_mat_arr, cluster_idx):
    assert(len(image_files) == len(truth_text_files))
    OUTPUT_TRUTH_DIR = "test_truth/"
    OUTPUT_PRED_DIR = "test_pred/"
    avg_time = 0
    for i in range(len(image_files)):
        image_file_name = image_files[i]
        text_file_name = truth_text_files[i]
        base_name = (ntpath.basename(image_file_name)).split('.')[0]
        output, time = recognize_image(image_file_name, features_mat_arr, lexicon, kmeans, cluster_idx)
        avg_time += time
        with open(OUTPUT_PRED_DIR + base_name + ".txt", "w") as file_handle:
            file_handle.write(output)
            file_handle.close()
        shutil.copyfile(text_file_name, OUTPUT_TRUTH_DIR + base_name + ".txt")
    print("Average time taken per file:", avg_time * 1.0 / len(image_files) )


def wb_generate_dataset(image_files, text_files):
    assert(len(image_files) == len(text_files))
    lexicon = load_if_exists(LEXICON_PATH, [])
    kmeans = load_if_exists(KMEANS_PATH, MiniBatchKMeans(n_clusters=CODEBOOK_SIZE, batch_size=500))
    features_mat = load_if_exists(FEATURES_MAT_PATH, [])
    processed_files = load_if_exists(PROCESSED_FILES_PATH, {})

    for i in range(len(image_files)):
        image_file_name = image_files[i]
        if image_file_name in processed_files:
            print("File already processed. Skipping..")
            continue
        text_file_name = text_files[i]
        text_file = open(text_file_name, encoding='utf-8')
        text_words = text_file.read().split(' ')
        image = io.imread(image_file_name)
        image = preprocess(image)
        image_lines = segment_lines(image)
        image_words = segment_words(image_lines)
        print("%s. # actual: %d. # segmented: %d." % ( image_file_name, len(text_words), len(image_words) ) )
        if len(text_words) != len(image_words):
            processed_files[image_file_name] = True
            continue
        features_mat, lexicon, kmeans = get_word_features(image_words, text_words, lexicon, kmeans, features_mat)
        processed_files[image_file_name] = True
        print("After %s: len(features_mat)=%d, len(lexicon)=%d" % (image_file_name, len(features_mat), len(lexicon)))
        if (i % 100) == 0:
            write_to_file(LEXICON_PATH, lexicon)
            write_to_file(FEATURES_MAT_PATH, features_mat)
            write_to_file(PROCESSED_FILES_PATH, processed_files)

    features_mat_arr = np.array(features_mat)
    print("GOING TO FIT!")
    kmeans.fit(features_mat_arr)
    labels = kmeans.predict(features_mat_arr)
    print("DONE FITTING!")
    cluster_idx = {i: np.where(labels == i)[0] for i in range(kmeans.n_clusters)}

    write_to_file(LEXICON_PATH, lexicon)
    write_to_file(KMEANS_PATH, kmeans)
    write_to_file(FEATURES_MAT_PATH, features_mat)
    write_to_file(PROCESSED_FILES_PATH, processed_files)
    return lexicon, kmeans, features_mat_arr, cluster_idx