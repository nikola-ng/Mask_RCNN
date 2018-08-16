import time

import cv2
import datetime
import imutils
import json
import os
import numpy as np

import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
sess = tf.Session(config=config)

from mrcnn import model as modellib, utils
from project.license_plate_location.car_exam import CarExamConfig, DEFAULT_LOGS_DIR, color_splash


def draw_poly(img, pt1, pt2, pt3, pt4):
    poly = np.array([[(pt1[0], pt1[1]), (pt2[0], pt2[1]), (pt3[0], pt3[1]), (pt4[0], pt4[1])]])
    cv2.fillPoly(img, poly, (255, 255, 255))
    return img


class InferenceConfig(CarExamConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

# Re-create model
model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir=DEFAULT_LOGS_DIR)
# Load weights
# weights_path = model.find_last()
weights_path = '../../weights/mask_rcnn_license_plate_0060.h5'
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

# load image
TEST_IMAGES_PATH = '/home/public/car_exam/gt_labeled_images/'
POST_PROCESSED_IMAGES_PATH = '/home/admin/github/chinese_ocr_keras/datasets/post_processed/'
PLATE_PATH = '/home/admin/github/chinese_ocr_keras/datasets/plate_ext/'
TEST_RESULT_PATH = '/home/admin/github/chinese_ocr_keras/datasets/evaluation_result_{}.csv'.format(
    model.log_dir.split(os.path.sep)[-1])

OFFSET = 0.2

KEY = {
    'license_plate': ['0111', '0112', '0164', '0322', '0323', '0348', '0351', '0352']
    # 'license_plate': ['0111', '0112', '0164']

}
pl = os.listdir(TEST_IMAGES_PATH)
pl_len = len(pl) // 3
pl = pl[:pl_len]

image_buff = []
image_buff_len = config.GPU_COUNT * config.IMAGES_PER_GPU

for p in pl:
    img_pl = os.listdir(TEST_IMAGES_PATH + '/' + p)
    for img_p in img_pl:
        if img_p.endswith('jpg'):
            for K in KEY['license_plate']:
                if K in img_p:
                    # Read image
                    img_path = TEST_IMAGES_PATH + '/' + p + '/' + img_p
                    image = cv2.imread(img_path)
                    h, w = image.shape[:2]
                    h_offset = int(h * 0.1)
                    w_offset = int(w * 0.1)
                    image = image[h_offset:h - h_offset, w_offset:w - w_offset]

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Detect objects
                    time1 = time.time()
                    # predict
                    # r = model.detect([image], verbose=1)[0]
                    r = model.detect([image], verbose=1)[0]
                    time2 = time.time()
                    if len(r['rois'] > 0):
                        process_time = time2 - time1
                        print('process time: {:.3}'.format(process_time))

                        roi_list = list(r['rois'])
                        score_list = list(r['scores'])
                        mask_list = cv2.split(r['masks'].astype(np.uint8))

                        # # Color splash
                        # splash = color_splash(image, r['masks'])
                        # splash = cv2.cvtColor(splash, cv2.COLOR_RGB2BGR)

                        # extract the biggest mask
                        if len(roi_list) > 1:
                            m_l = []
                            for i, mask in enumerate(mask_list):
                                m_l.append(np.count_nonzero(mask))
                            m_l = np.asarray(m_l)
                            max_mask_index = np.argmax(m_l)

                            # extract license plate
                            roi = roi_list[max_mask_index]
                            mask = mask_list[max_mask_index]
                        else:
                            roi = roi_list[0]
                            mask = mask_list[0]
                        y1, x1, y2, x2 = roi
                        y_offset = int((y2 - y1) * OFFSET)
                        x_offset = int((x2 - x1) * OFFSET * 1.5)
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
                        epsilon = 0.05 * cv2.arcLength(cnts[0], True)
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
                                cv2.imencode('.jpg', plate, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tofile(
                                    PLATE_PATH + '/{}'.format(p + '_' + img_p))
                                # cv2.drawContours(mask, approx, -1, (128, 128, 128), 2)
                                # cv2.imshow('plate', plate)
                                # cv2.waitKey(0)

                        # Save output
                        # cv2.imencode('.jpg', splash, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tofile(
                        #     POST_PROCESSED_IMAGES_PATH + '/{}'.format(p + '_' + img_p))
                        # cv2.imshow('Splash', imutils.resize(splash, width=800))
                        # cv2.waitKey(0)
                    else:
                        print('Failed to locate')
