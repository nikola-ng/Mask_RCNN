import cv2
import json
import os
import numpy as np
import shutil

from project.license_plate_location.build_json import ImageLabels

raw_images_path = '/home/public/car_exam/raw_images'
raw_marked_images_path = '/home/admin/github/Mask_RCNN/datasets/license_plate/raw'
json_label_path = '/home/admin/github/Mask_RCNN/datasets/license_plate/raw/via_region_data.json'

KEY = {
    'license_plate': ['0111', '0112', '0164', '0322', '0323', '0348', '0351', '0352']
    # 'license_plate': ['0111', '0112', '0164']

}

il = ImageLabels(json_label_path)

count = 0
# enter images paths of many cars
pl = os.listdir(raw_images_path)
for p in pl:
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

                                    raw_marked_image = raw_marked_images_path + '/' + \
                                                       plate_openalpr + '_{}'.format(img_p)
                                    shutil.copy(img_path, raw_marked_image)
                                    print(raw_marked_image)

                                    # dump label
                                    file_size = os.path.getsize(img_path)
                                    il.add_serial(plate_openalpr + '_{}'.format(img_p), file_size, 'polygon',
                                                  [pt1, pt2,pt3, pt4])

il.update()
il.close()
