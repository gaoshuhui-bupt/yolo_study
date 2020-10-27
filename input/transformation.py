#!/usr/bin/env python
#coding:utf-8
import sys
import time
import numpy as np
from PIL import Image, ImageEnhance
import cv2


def resize_short(img, target_size):
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(round(img.shape[1] * percent))
    resized_height = int(round(img.shape[0] * percent))
    img = cv2.resize(img, (resized_width, resized_height), interpolation = cv2.INTER_LINEAR)
    return img


def crop_image(img, target_size, center=True):
    width, height = img.shape[1], img.shape[0]
    size = target_size
    if center == True:
        w_start = (width - size) // 2
        h_start = (height - size) // 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img[h_start:h_end, w_start:w_end]
    return img


def random_crop(img, size, scale=[0.08,1.0], ratio=[3./4.,4./3.], flip=True, rot90=True):
    aspect_ratio = np.sqrt(np.random.uniform(*ratio))
    if np.random.rand() < 0.5:
        aspect_ratio = 1.0 / aspect_ratio
    ws = 1. * aspect_ratio
    hs = 1. / aspect_ratio
    
    imgsize = (img.shape[1], img.shape[0])
    imgarea = imgsize[0] * imgsize[1]
    target_area = imgarea * np.random.uniform(*scale)
    target_size = np.sqrt(target_area)
    w = min(imgsize[0], max(1, int(0.5 + target_size * ws)))
    h = min(imgsize[1], max(1, int(0.5 + target_size * hs)))
    
    i = np.random.randint(0, imgsize[0] - w + 1)
    j = np.random.randint(0, imgsize[1] - h + 1)
    img = img[j:j+h, i:i+w]
    img = cv2.resize(img, (size, size), interpolation = cv2.INTER_LINEAR)
    
    if flip:
        if np.random.rand() < 0.5:
            img = np.fliplr(img)
    if rot90:
        p = np.random.rand()
        if p < 0.79:   # +0
            pass
        elif p < 0.86: # +90
            img = np.rot90(img)
        elif p < 0.93: # +180
            img = np.rot90(img, 2)
        elif p < 1.0: # +270
            img = np.rot90(img, 3)
    
    return img


def rotate_image(img):
    if np.random.rand() < 0.8:
        return img
    # very slow
    angle = np.random.randint(-15, 16)
    if angle > 3 or angle < -3:
        img = Image.fromarray(img)
        img = img.rotate(angle, expand=1)
        img = np.array(img, dtype='uint8')
    return img


def distort_color(img, color_pca=True):
    def random_brightness(img, lower=0.5, upper=1.5):
        img = np.clip(img, 0.0, 1.0)
        e = np.random.uniform(lower, upper)
        # zero = np.zeros([1] * len(img.shape), dtype=img.dtype)
        return img * e # + zero * (1.0 - e)

    def random_contrast(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        gray = np.mean(img[0]) * 0.299 + np.mean(img[1]) * 0.587 + np.mean(img[2]) * 0.114
        mean = np.ones([1] * len(img.shape), dtype=img.dtype) * gray
        return img * e + mean * (1.0 - e)

    def random_color(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        gray = img[0] * 0.299 + img[1] * 0.587 + img[2] * 0.114
        gray = np.expand_dims(gray, axis=0)
        return img * e + gray * (1.0 - e)

    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)

    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)

    if color_pca:
        eigvec = np.array([ [-0.5675,  0.7192,  0.4009],
                            [-0.5808, -0.0045, -0.8140],
                            [-0.5836, -0.6948,  0.4203] ], dtype='float32')
        alpha = (np.random.rand(3) * 0.1).astype('float32')
        eigval = np.array([0.2175, 0.0188, 0.0045], dtype='float32')
        rgb = np.sum(eigvec * alpha * eigval, axis=1)
        img += rgb.reshape([3, 1, 1])
    
    #img = np.clip(img, 0.0, 1.0)

    return img


img_mean = np.array([0.485, 0.456, 0.406], dtype='float32').reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225], dtype='float32').reshape((3, 1, 1))

def std_image(img):
    img -= img_mean
    img *= (1.0 / img_std)
    return img


