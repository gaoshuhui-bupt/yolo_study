#!/usr/bin/env python
#coding:utf-8
import sys
import os
import time
import pybase64 as base64
import random
import numpy as np
import multiprocessing as mps
import ctypes as ctp
from turbojpeg import TurboJPEG, TJPF_RGB
import cv2
from PIL import Image, ImageEnhance
from .transformation import rotate_image, random_crop, distort_color
from .transformation import resize_short, crop_image, std_image

import torch
import horovod.torch as hvd


turbojpeg = TurboJPEG('./input/libturbojpeg.so.0.2.0')


def _process_image(img, mode, color_jitter, rotate, data_dim):
    try:
        # don't support the jpeg image with a CMYK color space
        img = turbojpeg.decode(img, pixel_format=TJPF_RGB)
        if not (img is not None and len(img.shape) == 3 and img.shape[-1] == 3
            and img.max() > 0 and img.shape[0] > 5 and img.shape[1] > 5):
            print('image decode error! (from turbojpeg)')
            return None
    except:
        img = cv2.imdecode(np.asarray(bytearray(img),dtype="uint8"), cv2.IMREAD_COLOR)
        if not (img is not None and len(img.shape) == 3 and img.shape[-1] == 3
            and img.max() > 0 and img.shape[0] > 5 and img.shape[1] > 5):
            print('image decode error! (from cv2)')
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if mode == 'train':
        if rotate:
            img = rotate_image(img)
        img = random_crop(img, data_dim)
        img = img.astype('float32').transpose((2, 0, 1)) * (1.0 / 255)
        if color_jitter:
            img = distort_color(img)
    else:
        img = resize_short(img, target_size=256)
        img = crop_image(img, target_size=data_dim, center=True)
        img = img.astype('float32').transpose((2, 0, 1)) * (1.0 / 255)

    img = std_image(img)

    return img


def _process_line(line, token_idx, mode, color_jitter, rotate, data_dim):
    tokens_data = line.strip().split('\t')
    img = base64.b64decode(tokens_data[token_idx[0]].replace('-', '+').replace('_', '/'))
    img = _process_image(img, mode, color_jitter, rotate, data_dim)
    if img is None:
        return []
    label = int(tokens_data[token_idx[1]])
    sample = [img, label]
    return [sample]


def _reader_creator(batch_size, data_dir, file_list, token_idx, data_dim,
                    mode,
                    shuffle=False,
                    color_jitter=False,
                    rotate=False,
                    func_process_line=None):
    
    if not data_dir.endswith('/'):
        data_dir += '/'
    
    with open(file_list) as flist:
        file_names_list = [line.strip() for line in flist]
    print('parts,', len(file_names_list))

    file_names_q = mps.Queue()
    # 在多机间保证图片不会重复
    if hvd.size() > 1:
        random.seed(123)
        i = 0
        for j in range(200):
            names_list = file_names_list[:]
            if shuffle:
                random.shuffle(names_list)
            if i == 0:
                print(hvd.size(), hvd.rank(), names_list[:2])
            for name in names_list:
                if i % hvd.size() == hvd.rank():
                    file_names_q.put(name)
                i += 1
        print(hvd.size(), hvd.rank(), file_names_q.get())

    def _get_a_name_from_q():
        if file_names_q.empty():
            names_list = file_names_list[:]
            if shuffle:
                random.shuffle(names_list)
            for name in names_list:
                file_names_q.put(name)
        return file_names_q.get()

    if mode == 'train':
        process_num = 8
        queue_batch_size = 25000
        shuffle_size_of_bs = 10000
    elif mode == 'val':
        process_num = 16
        queue_batch_size = 10000
        shuffle_size_of_bs = 1000
    queue_batch_size = int(queue_batch_size/batch_size)
    shuffle_size_of_bs = int(shuffle_size_of_bs/batch_size)

    # multiprocessing.Queue is very slow !!
    # use shared memory
    batchs_q = mps.Queue(queue_batch_size + 1)
    batchs_q_r = mps.Queue(queue_batch_size + 1)
    data_batchs_img = []
    data_batchs_lab = []
    for i in range(queue_batch_size):
        data_batchs_img.append(mps.Array(ctp.c_byte, batch_size*3*data_dim*data_dim*4))
        data_batchs_lab.append(mps.Array(ctp.c_byte, batch_size*8))
        batchs_q_r.put(i)

    should_stop = mps.Value('i', 0)

    def _read_img_jpg_to_q():
        shuffle_lines = []
        samples = []
        while True:
            while len(shuffle_lines) < batch_size * shuffle_size_of_bs:
                name = _get_a_name_from_q()
                with open(data_dir + name) as f:
                    shuffle_lines += f.readlines()
            
            while len(samples) < batch_size:
                if shuffle:
                    i = np.random.randint(0, len(shuffle_lines))
                    line = shuffle_lines[i]
                    shuffle_lines = shuffle_lines[:i] + shuffle_lines[i+1:]
                else:
                    line = shuffle_lines[0]
                    shuffle_lines = shuffle_lines[1:]
                line_samps = func_process_line(line, token_idx, mode, color_jitter, rotate, data_dim)
                samples += line_samps
            
            imgs = np.array([i[0] for i in samples[:batch_size]])
            labs = np.array([i[1] for i in samples[:batch_size]])
            samples = samples[batch_size:]
            
            if not(imgs.dtype=='float32' and imgs.shape==(batch_size, 3, data_dim, data_dim)
                and imgs.nbytes==batch_size*3*data_dim*data_dim*4):
                print('imgs type error!')
                continue
            if not(labs.dtype=='int64' and labs.shape==(batch_size,)
                and labs.nbytes==batch_size*8):
                print('labs type error!')
                continue

            idx = batchs_q_r.get()
            ctp.memmove(data_batchs_img[idx].get_obj(), imgs.ctypes.data, imgs.nbytes)
            ctp.memmove(data_batchs_lab[idx].get_obj(), labs.ctypes.data, labs.nbytes)
            batchs_q.put(idx)

            if should_stop.value == 1:
                break

    p_list = []
    for i in range(process_num):
        p = mps.Process(target=_read_img_jpg_to_q, args=())
        p.daemon = True
        p.start()
        p_list.append(p)

    def reader():
        if mode == 'train':
            py_reader_capacity = 16
        elif mode == 'val':
            py_reader_capacity = 8
        idx_being_used = []
        while True:
            idx = batchs_q.get()
            imgs = np.frombuffer(data_batchs_img[idx].get_obj(), dtype='float32').reshape(
                [batch_size, 3, data_dim, data_dim])
            labs = np.frombuffer(data_batchs_lab[idx].get_obj(), dtype='int64').reshape(
                [batch_size])
            idx_being_used.append(idx)
            if len(idx_being_used) > py_reader_capacity:
                batchs_q_r.put(idx_being_used[0])
                idx_being_used = idx_being_used[1:]
            yield torch.from_numpy(imgs), torch.from_numpy(labs)

    def reset():
        should_stop.value = 1
        if batchs_q_r.qsize() < process_num:
            for i in range(process_num):
                idx = batchs_q.get()
                batchs_q_r.put(idx)
        for p in p_list:
            p.join()

    return reader, reset
    

def train(batch_size, data_dim, data_dir, file_list, token_idx,
            func_reader_creator=_reader_creator,
            func_process_line=_process_line):
    """Args:
        batch_size: batch size per GPU.
        data_dim: e.g. 224.
        data_dir: root directory or image parts.
        file_list: File list for hdfs files(parts).
        token_idx: [3,1] means line.strip().split('\t')[3] is img_b64 and ..[1] is label.
    Returns:
        images: Batches of images. [batch_size, 3, data_dim, data_dim] (i.e. NCHW).
        labels: Batches of labels. [batch_size, 1].
        train_py_reader: to call train_py_reader.reset().
        _reset: to close background progresses decoding images.
    """
    _reader, _reset = func_reader_creator(batch_size, data_dir, file_list, token_idx, data_dim, 'train',
            shuffle=True, color_jitter=True, rotate=False, func_process_line=func_process_line)
    
    return _reader, _reset


def val(batch_size, data_dim, data_dir, file_list, token_idx,
            func_reader_creator=_reader_creator,
            func_process_line=_process_line):
    _reader, _reset = func_reader_creator(batch_size, data_dir, file_list, token_idx, data_dim, 'val',
            shuffle=False, color_jitter=False, rotate=False, func_process_line=func_process_line)
    
    return _reader, _reset

