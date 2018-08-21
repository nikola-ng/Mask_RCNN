import cv2
import json
import os
from sklearn.model_selection import train_test_split

from tools.build_json import ImageLabels

raw_images_path = '/home/public/car_exam/raw_images'
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
pl = os.listdir(raw_images_path)

pl_train, pl_val = train_test_split(pl, test_size=0.4, random_state=42)
pl_val, pl_test = train_test_split(pl_val, test_size=0.75, random_state=42)

p_com = [train_marked_images_path, val_marked_images_path, test_marked_images_path]
pl_com = [pl_train, pl_val, pl_test]
il_com = [il_train, il_val, il_test]

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
                                        # shutil.copy(img_path, raw_marked_image)

                                        # fix image error
                                        img = cv2.imread(img_path)
                                        cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 100])[1] \
                                            .tofile(raw_marked_image)

                                        print(raw_marked_image)

                                        # dump label
                                        file_size = os.path.getsize(raw_marked_image)
                                        il_com[i].add_serial(plate_openalpr + '_{}'.format(img_p), file_size, 'polygon',
                                                             [pt1, pt2, pt3, pt4])
                                else:
                                    raw_marked_image = unmarked_images_path + '/' + \
                                                       str(count).zfill(6) +\
                                                       '_{}'.format(img_p)
                                    img = cv2.imread(img_path)
                                    cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 100])[1] \
                                        .tofile(raw_marked_image)
                                    print(raw_marked_image)

    il_com[i].update()
    il_com[i].close()
