import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate, resize, rescale
import deskew

LINE_HEIGHT = 40
BASELINE_THICKNESS = 2
BASELINE_SCAFFOLD = 5


def highlight(img_like, places, axis=0):
    img = np.zeros_like(img_like)
    if axis==0:
        img[places, :] = 1
    else:
        img[:, places] = 1
    return img


def show_images(images):
    for img in images:
        plt.imshow(img, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

def deskew_image(img):
    angle = deskew.determine_skew(img)
    img = rotate(img, angle, resize=True, mode='constant', cval=1)
    return img

def image_autocrop(img):
    # TODO: refactor this.
    vertical_projection = np.sum(img, axis=0)
    height, width = img.shape
    start = 0
    while vertical_projection[start] == 0:
        start += 1
    end = width-1
    while vertical_projection[end] == 0:
        end -= 1
    img = img[:, start:end+1]

    horizontal_projection = np.sum(img, axis=1)
    start = 0
    while horizontal_projection[start] == 0:
        start += 1
    end = height-1
    while horizontal_projection[end] == 0:
        end -= 1
    img = img[start:end+1, :]
    return img

def get_bounds(img):
    # TODO: refactor this.
    vertical_projection = np.sum(img, axis=0)
    height, width = img.shape
    start = 0
    while vertical_projection[start] == 0:
        start += 1
    end = width-1
    while vertical_projection[end] == 0:
        end -= 1

    horizontal_projection = np.sum(img, axis=1)
    start_h = 0
    while horizontal_projection[start_h] == 0:
        start_h += 1
    end_h = height-1
    while horizontal_projection[end_h] == 0:
        end_h -= 1
    return start_h, end_h, start, end


def debug_draw(red, blue=None, green=None):
    if green is None:
        green = np.zeros((red.shape[0], red.shape[1]))
    if blue is None:
        blue = np.zeros((red.shape[0], red.shape[1]))
    new_img = np.zeros((red.shape[0], red.shape[1], 3))
    new_img[:, :, 0] = red
    new_img[:, :, 1] = green
    new_img[:, :, 2] = blue
    show_images([new_img])