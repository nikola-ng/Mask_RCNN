import time

import cv2
import datetime
import imutils
import json
import os
import numpy as np

from mrcnn import model as modellib, utils
from project.license_plate_location.car_exam import LicensePlateConfig, DEFAULT_LOGS_DIR, color_splash

# run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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
POST_PROCESSED_IMAGES_PATH = '../../datasets/license_plate/post_processed/'
PLATE_PATH = '../../datasets/license_plate/plate/'
TEST_JSON_PATH = '../../datasets/license_plate/test/via_region_data.json'
TEST_RESULT_PATH = '../../datasets/license_plate/evaluation_result_{}.csv'.format(
    model.log_dir.split(os.path.sep)[-1])

JS_FILE = open(TEST_JSON_PATH)
JS = json.loads(JS_FILE.read())
JS_NO_SIZE = {}
for key in JS:
    JS_NO_SIZE[key.split('.jpg')[0]] = JS[key]
JS = JS_NO_SIZE
OFFSET = 0.2

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
        if len(r['rois'] > 0):
            process_time = time2 - time1
            print('process time: {:.3}'.format(process_time))

            roi_list = list(r['rois'])
            score_list = list(r['scores'])
            mask_list = cv2.split(r['masks'].astype(np.uint8))

            serial_name = p.split('.jpg')[0]
            region_list = JS[serial_name]['regions']

            # # Color splash
            splash = color_splash(image, r['masks'])
            splash = cv2.cvtColor(splash, cv2.COLOR_RGB2BGR)

            for region in region_list:
                x_set = region['shape_attributes']['all_points_x']
                y_set = region['shape_attributes']['all_points_y']
                pt_set = [(x, y) for x, y in zip(x_set, y_set)]
                raw_area = draw_poly(raw_area, pt_set[0], pt_set[1], pt_set[2], pt_set[3])

                cv2.line(splash, pt_set[0], pt_set[1], (0, 255, 0), 2)
                cv2.line(splash, pt_set[1], pt_set[2], (0, 255, 0), 2)
                cv2.line(splash, pt_set[2], pt_set[3], (0, 255, 0), 2)
                cv2.line(splash, pt_set[3], pt_set[0], (0, 255, 0), 2)

            for mask in mask_list:
                predicted_area = predicted_area + mask
            predicted_area[predicted_area > 0] = 255
            predicted_area = predicted_area.astype(np.uint8)

            iou = cv2.bitwise_and(predicted_area, raw_area)
            iou_score = np.count_nonzero(iou) / np.count_nonzero(raw_area)
            mean_score = np.mean(score_list)
            print('IoU: {}, Confidence: {}'.format(iou_score, mean_score))
            RESULT.write('{},{},{},{}\n'.format(p, iou_score, mean_score, process_time))

            # extract license plate
            for roi, mask in zip(roi_list, mask_list):
                y1, x1, y2, x2 = roi
                y_offset = int((y2 - y1) * OFFSET)
                x_offset = int((x2 - x1) * OFFSET)
                y1 = y1 - y_offset
                y2 = y2 + y_offset
                x1 = x1 - x_offset
                x2 = x2 + x_offset

                y1 = 0 if y1 < 0 else y1
                y2 = image.shape[0] if y2 > image.shape[0] else y2
                x1 = 0 if x1 < 0 else x1
                x2 = image.shape[1] if x2 > image.shape[1] else x2

                plate = image[y1:y2, x1:x2]
                mask = mask[y1:y2, x1:x2]
                mask = cv2.dilate(mask, None, iterations=4)
                # noticing it's APPROX_NONE, check out
                # https://stackoverflow.com/questions/41576815/drawing-contours-using-cv2-approxpolydp-in-python
                cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
                epsilon = 0.1 * cv2.arcLength(cnts[0], True)
                if len(cnts) > 0:
                    approx = cv2.approxPolyDP(cnts[0], epsilon, True)
                    if len(approx) >= 4:
                        # cv2.imshow('plate_pre', plate)
                        # cv2.waitKey(0)
                        approx = approx[:, 0, :]
                        ul = approx[np.argmin(approx[:, 0] + approx[:, 1])]
                        lr = approx[np.argmax(approx[:, 0] + approx[:, 1])]
                        ur = approx[np.argmin(np.diff(approx, axis=1))]
                        ll = approx[np.argmax(np.diff(approx, axis=1))]
                        approx = np.array([ul, ur, lr, ll]).astype(np.float32)
                        h, w = plate.shape[:2]
                        post_perspective = np.array([[0, 0], [w, 0], [w, h], [0, h]]).astype(np.float32)
                        M = cv2.getPerspectiveTransform(approx, post_perspective)
                        plate = cv2.warpPerspective(plate, M, (w, h), flags=cv2.INTER_LANCZOS4,
                                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                        plate = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)
                        cv2.imencode('.jpg', plate, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tofile(PLATE_PATH + '/{}'.format(p))
                        # cv2.drawContours(mask, approx, -1, (128, 128, 128), 2)
                        # cv2.imshow('plate', plate)
                        # cv2.waitKey(0)

                # Save output
                cv2.imencode('.jpg', splash, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tofile(POST_PROCESSED_IMAGES_PATH + '/{}'.format(p))
                # cv2.imshow('Splash', imutils.resize(splash, width=800))
                # cv2.waitKey(0)
        else:
            print('Failed to locate')
            RESULT.write('{},0,0,0\n'.format(p))

RESULT.close()
JS_FILE.close()
