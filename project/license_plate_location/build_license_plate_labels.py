import cv2
import json
import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

from project.license_plate_location.build_json import ImageLabels

raw_images_path = '/home/public/car_exam/raw_images'
train_marked_images_path = '/home/admin/github/Mask_RCNN/datasets/license_plate/train'
val_marked_images_path = '/home/admin/github/Mask_RCNN/datasets/license_plate/val'
json_label_path_train = '/home/admin/github/Mask_RCNN/datasets/license_plate/train/via_region_data.json'
json_label_path_val = '/home/admin/github/Mask_RCNN/datasets/license_plate/val/via_region_data.json'

KEY = {
    'license_plate': ['0111', '0112', '0164', '0322', '0323', '0348', '0351', '0352']
    # 'license_plate': ['0111', '0112', '0164']
}

il_train = ImageLabels(json_label_path_train)
il_val = ImageLabels(json_label_path_val)

count = 0
# enter images paths of many cars
pl = os.listdir(raw_images_path)

pl_train, pl_val = train_test_split(pl, test_size=0.25, random_state=42)

p_com = [train_marked_images_path, val_marked_images_path]
pl_com = [pl_train, pl_val]
il_com = [il_train, il_val]

for i, c in enumerate(pl_com):
    for p in pl_com[i]:
        imgs_path = raw_images_path + '/' + p
        if os.path.isdir(imgs_path):
            # enter images path of a single car
            imgs_pl = os.listdir(imgs_path)
            license_no = []
            for img_p in imgs_pl:
                img_path = imgs_path + '/' + img_p

                # loop over every image that satisfies the requirements
                if img_path.lower().endswith(
                        '.jpg') or img_path.lower().endswith(
                    '.jpeg') or img_path.lower().endswith(
                    '.png'):
                    for K in KEY['license_plate']:
                        if K in img_path:
                            json_path = img_path.split('.jpg')[0] + '.json'
                            if os.path.exists(img_path.split('.jpg')[0] + '.json'):
                                count += 1

                                # start processing
                                js = json.loads(open(json_path).read())

                                result = js['results']
                                if len(result) > 0:
                                    for r in result:
                                        coords = r['coordinates']
                                        coords = [(co['x'], co['y']) for i, co in enumerate(coords)]
                                        plate_openalpr = r['plate']

                                        pt1 = (coords[0][0], coords[0][1])
                                        pt2 = (coords[1][0], coords[1][1])
                                        pt3 = (coords[2][0], coords[2][1])
                                        pt4 = (coords[3][0], coords[3][1])

                                        raw_marked_image = p_com[i] + '/' + \
                                                           plate_openalpr + '_{}'.format(img_p)
                                        shutil.copy(img_path, raw_marked_image)
                                        print(raw_marked_image)

                                        # dump label
                                        file_size = os.path.getsize(img_path)
                                        il_com[i].add_serial(plate_openalpr + '_{}'.format(img_p), file_size, 'polygon',
                                                            [pt1, pt2, pt3, pt4])

    il_com[i].update()
    il_com[i].close()
