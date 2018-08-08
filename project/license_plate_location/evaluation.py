import time

import cv2
import datetime
import imutils
import json
import os
import numpy as np

from mrcnn import model as modellib, utils
from project.license_plate_location.license_plate import LicensePlateConfig, DEFAULT_LOGS_DIR, color_splash

# run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def draw_poly(img, pt1, pt2, pt3, pt4):
    poly = np.array([[(pt1[0], pt1[1]), (pt2[0], pt2[1]), (pt3[0], pt3[1]), (pt4[0], pt4[1])]])
    cv2.fillPoly(img, poly, (255, 255, 255))
    return img


class InferenceConfig(LicensePlateConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

# Re-create model
model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir=DEFAULT_LOGS_DIR)
# Load weights
weights_path = model.find_last()
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

# load image
TEST_IMAGES_PATH = '../../datasets/license_plate/test/'
TEST_JSON_PATH = '../../datasets/license_plate/test/via_region_data.json'
TEST_RESULT_PATH = '../../datasets/license_plate/evaluation_result_{}.csv'.format(
    model.log_dir.split(os.path.sep)[-1])

JS_FILE = open(TEST_JSON_PATH)
JS = json.loads(JS_FILE.read())
JS_NO_SIZE = {}
for key in JS:
    JS_NO_SIZE[key.split('.jpg')[0]] = JS[key]
JS = JS_NO_SIZE

RESULT = open(TEST_RESULT_PATH, 'w')
pl = os.listdir(TEST_IMAGES_PATH)
for p in pl:
    if p.endswith('jpg'):
        # Read image
        img_path = TEST_IMAGES_PATH + '/' + p
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        raw_area = np.zeros_like(gray_image, dtype=np.uint8)
        predicted_area = np.zeros_like(gray_image, dtype=np.uint32)
        # Detect objects
        time1 = time.time()
        r = model.detect([image], verbose=1)[0]
        time2 = time.time()
        print('process time: {:.3}'.format((time2 - time1) / len(r['rois'])))

        roi_list = list(r['rois'])
        score_list = list(r['scores'])
        mask_list = cv2.split(r['masks'].astype(np.uint8))

        serial_name = p.split('.jpg')[0]
        region_list = JS[serial_name]['regions']

        for region in region_list:
            x_set = region['shape_attributes']['all_points_x']
            y_set = region['shape_attributes']['all_points_y']
            pt_set = [(x, y) for x, y in zip(x_set, y_set)]
            raw_area = draw_poly(raw_area, pt_set[0], pt_set[1], pt_set[2], pt_set[3])

        for mask in mask_list:
            predicted_area = predicted_area + mask
        predicted_area[predicted_area > 0] = 255
        predicted_area = predicted_area.astype(np.uint8)

        iou = cv2.bitwise_and(predicted_area, raw_area)
        iou_score = np.count_nonzero(iou) / np.count_nonzero(raw_area)
        mean_score = np.mean(score_list)
        print('IoU: {}, Confidence: {}'.format(iou_score, mean_score))
        RESULT.write('{},{},{}'.format(p, iou_score, mean_score))
        # # Color splash
        # splash = color_splash(image, r['masks'])
        # splash = cv2.cvtColor(splash, cv2.COLOR_RGB2BGR)
        # Show output
        # cv2.imshow('Splash', imutils.resize(splash, width=800))
        # cv2.waitKey(0)
RESULT.close()
JS_FILE.close()
