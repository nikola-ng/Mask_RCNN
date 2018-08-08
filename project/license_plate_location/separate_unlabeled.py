import cv2
import json
import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

from project.license_plate_location.build_json import ImageLabels

train_marked_images_path = '/home/admin/github/Mask_RCNN/datasets/license_plate/train'
val_marked_images_path = '/home/admin/github/Mask_RCNN/datasets/license_plate/val'
test_marked_images_path = '/home/admin/github/Mask_RCNN/datasets/license_plate/test'
unmarked_images_path = '/home/admin/github/Mask_RCNN/datasets/license_plate/unlabeled'
json_label_path_train = '/home/admin/github/Mask_RCNN/datasets/license_plate/train/via_region_data.json'
json_label_path_val = '/home/admin/github/Mask_RCNN/datasets/license_plate/val/via_region_data.json'
json_label_path_test = '/home/admin/github/Mask_RCNN/datasets/license_plate/test/via_region_data.json'

KEY = {
    'license_plate': ['0111', '0112', '0164', '0322', '0323', '0348', '0351', '0352']
    # 'license_plate': ['0111', '0112', '0164']
}

il_train = ImageLabels(json_label_path_train)
il_val = ImageLabels(json_label_path_val)
il_test = ImageLabels(json_label_path_test)

count = 0
# enter images paths of many cars
pl = os.listdir(unmarked_images_path)

# train 6, val 1, test 3
pl_train, pl_val = train_test_split(pl, test_size=0.4, random_state=42)
pl_val, pl_test = train_test_split(pl_val, test_size=0.75, random_state=42)

p_com = [train_marked_images_path, val_marked_images_path, test_marked_images_path]
pl_com = [pl_train, pl_val, pl_test]
il_com = [il_train, il_val, il_test]

js = json.loads(open('/home/admin/github/Mask_RCNN/datasets/license_plate/unlabeled/via_region_data.json').read())
for key in js:
    name = key.split('.jpg')[0]
    for p, pl, il in zip(p_com, pl_com, il_com):
        for pl_data in pl:
            if name in pl_data:
                il.json_data[key] = js[key]
                shutil.copy(unmarked_images_path + '/{}.jpg'.format(name), p)

for il in il_com:
    il.update()
    il.close()
