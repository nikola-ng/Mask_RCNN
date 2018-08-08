import cv2
import numpy as np


def normalize(img, mask):
    plate_list = []

    # extract every one of the license plates
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            empty = np.zeros_like(mask)
            cv2.drawContours(empty, [c], 0, (255, 255, 255), -1)
            x, y, w, h = cv2.minAreaRect(c)
            plate = img[y:y + h, x:x + w]

            # find the approximate


