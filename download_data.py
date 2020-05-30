import os

import numpy as np
import urllib.request as urllib
import cv2
import json

from config import IMAGE_DIM
from progress_bar import ProgressBar

def get_image(url):
    response = urllib.urlopen(url)
    image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def get_resized_image(url):
    return cv2.resize(get_image(url), IMAGE_DIM)

artist = 'Seb McKinnon'

path = 'images/' + artist

if not os.path.exists(path):
    os.mkdir(path)

with open('./data/all-cards-20200529172208.json', encoding='utf-8') as f:
    data = json.load(f)

    bar = ProgressBar(len(data))

    for card_obj in data:
        if 'artist' in card_obj and card_obj['artist'] == artist and 'image_uris' in card_obj and 'art_crop' in card_obj['image_uris']:
            img = get_resized_image(card_obj['image_uris']['art_crop'])
            cv2.imwrite(path + "/" + card_obj['name'] + ".png", img)
        bar.increment()