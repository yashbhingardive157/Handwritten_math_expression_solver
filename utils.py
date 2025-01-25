from scipy import ndimage
import cv2 as cv
import numpy as np
import math

def getBestShift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)

    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)

    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv.warpAffine(img, M, (cols, rows))
    return shifted


def prep_img(img):
    img = cv.resize(255 - img, (28, 28))

    thresh, img = cv.threshold(img, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    while np.sum(img[0]) == 0:
        img = img[1:]

    while np.sum(img[:, 0]) == 0:
        img = np.delete(img, 0, 1)

    while np.sum(img[-1]) == 0:
        img = img[:-1]

    while np.sum(img[:, -1]) == 0:
        img = np.delete(img, -1, 1)

    rows, cols = img.shape

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
  
        img = cv.resize(img, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))

        img = cv.resize(img, (cols, rows))

    cols_padding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rows_padding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    img = np.pad(img, (rows_padding, cols_padding), 'constant')

    shiftx, shifty = getBestShift(img)
    img = shift(img, shiftx, shifty)
    return img
