import cv2
import imutils
import json
import os
import numpy as np

from mrcnn import model as modellib, utils
from project.license_plate_location.license_plate import LicensePlateConfig, DEFAULT_LOGS_DIR, color_splash

# run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def draw_poly(img, pt1, pt2, pt3, pt4):
    poly = np.array([[[pt1[0], pt1[1]], [pt2[0], pt2[1]], [pt3[0], pt3[1]], [pt4[0], pt4[1]]]], dtype=np.uint8)
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
JS = json.loads(open(TEST_JSON_PATH).read())
pl = os.listdir(TEST_IMAGES_PATH)
for p in pl:
    if p.endswith('jpg'):
        # Read image
        img_path = TEST_IMAGES_PATH + '/' + p
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        raw_area = np.zeros_like(gray_image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]

        roi_list = r['rois']
        score_list = r['scores']
        mask_list = r['masks']

        image_size = str(os.path.getsize(img_path))
        serial_name = p + image_size
        region_list = JS[serial_name]['regions']

        for region in region_list:
            x_set = region['shape_attributes']['all_points_x']
            y_set = region['shape_attributes']['all_points_y']
            pt_set = [(x, y) for x, y in zip(x_set, y_set)]
            raw_area = draw_poly(raw_area, pt_set[0], pt_set[1], pt_set[2], pt_set[3])

        for roi, score, mask in zip(roi_list, score_list, mask_list):
            iou = cv2.bitwise_and(mask, raw_area)
            iou_score = np.count_nonzero(iou) / np.count_nonzero(raw_area)
        # # Color splash
        # splash = color_splash(image, r['masks'])
        # splash = cv2.cvtColor(splash, cv2.COLOR_RGB2BGR)
        # Show output
        print()
        # cv2.imshow('Splash', imutils.resize(splash, width=800))
        # cv2.waitKey(0)
