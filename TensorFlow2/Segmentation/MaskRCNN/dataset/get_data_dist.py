from pycocotools.coco import COCO
from collections import defaultdict, Counter
from pprint import pprint
import random
import copy
import os
import shutil
import re

# same sequence
random.seed(17)

# paths
annFile='/shared/data3/annotations/instances_train2017.json'

# initialize COCO api for instance annotations
coco=COCO(annFile)
images = coco.dataset['images']
image_id_to_detail_map = {}
for image in images:
    image_id_to_detail_map[image['id']] = image

categories = coco.dataset['categories']

cats = coco.loadCats(coco.getCatIds())

class_to_images = defaultdict(list)

# get image ids for each class
for cat in cats:
    obj_id = cat['id']
    img_ids = coco.getImgIds(catIds=[obj_id])
    if img_ids:
        class_to_images[obj_id].extend(img_ids)

target_images_per_class = 1500 # roughly calculated as 120000/80

# sampling
for k, v in class_to_images.items():
    print(k, len(v))

