import random
from itertools import cycle

import cv2
import imutils
import numpy as np
import torch


def _resize(image):
    h,w,c = image.shape
    min_dim = min(h, w)
    img_t = image[:, int(w/2-min_dim/2):int(w/2+min_dim/2)]
    img_t = cv2.resize(img_t, (300, 300))
    return img_t


def _histogram_equalization(image):
    cla = image.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    for i in range(2):
        cla[:, :, i] = clahe.apply(image[:, :, i])
    return cla 


def _flip(image):
    flipcode = random.randint(-1, 1)
    image = cv2.flip(image, flipcode)
    return image


def _rotate(image):
    angle = random.randint(-5, 5)
    rotated = imutils.rotate(image, angle)
    return rotated


class preproc(object):

    def __init__(self):
        pass

    def __call__(self, image, targets, phase):
        image_t = _resize(image)
        image_t = _histogram_equalization(image_t)
        if phase == 'training' or phase == 'validate':
            labels = torch.tensor(
                np.array([targets]), dtype=torch.long)
            if phase == 'training':
                if bool(random.getrandbits(1)):
                    image_t = _flip(image_t)
                if bool(random.getrandbits(1)):
                    image_t = _rotate(image_t)
        else:
            labels = targets
        image_t = (image_t/255.0).astype(np.float32)
        return image_t, labels
