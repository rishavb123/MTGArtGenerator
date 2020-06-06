import os

import time
import datetime
import numpy as np
import cv2

from config import PROGRESS_BAR

def load_data(image_dir, should_log=True):
    images = []

    ls = os.listdir(image_dir + '/color/')
    l = len(ls)

    def log(*args, **kwargs):
        if should_log:
            print(*args, **kwargs)

    log('Loading data from', '"' + image_dir + '"', '. . .')

    cur_time = time.time()

    progress_bar_width = PROGRESS_BAR["width"]
    progress_char = PROGRESS_BAR["positive"]
    empty_char = PROGRESS_BAR["negative"]

    for i in range(l):
        w = i * progress_bar_width // l
        s = int(time.time() - cur_time)
        log('[' + progress_char * w + empty_char * (progress_bar_width - w) + ']', i, '/', l, datetime.timedelta(seconds=s), end='\r')

        fname = ls[i]
        images.append(cv2.imread(image_dir + fname))

    log('[' + progress_char * progress_bar_width + ']', l, '/', l)
    log('Finished loading data - Took', int(time.time() - cur_time), 'seconds')
    return np.array(images, np.float32)

def load_data_by_author(author, should_log=True):
    return load_data('./images/' + author, should_log=should_log)

def normalize(faces, output_range=(-1, 1), input_range=(0, 255)):
    return (faces - input_range[0]) * (output_range[1] - output_range[0]) / (input_range[1] - input_range[0]) + output_range[0]