import cv2
import numpy as np
import json
import random
from detection.canny_hough.painting_detection import get_contours
from math import sqrt

from detection.threshold_ccl.threshold_ccl import run_frame


def crop(img, box):
    x_left = box['y']
    x_right = x_left + box['height']
    y_left = box['x']
    y_right = y_left + box['width']

    return img[x_left:x_right, y_left:y_right, :]


def rect(frame, bboxes):
    out = frame.copy()
    for bbox in bboxes:
        copy = frame.copy()
        src_pts = []
        dst_pts = [
            [bbox['x'], bbox['y']],
            [bbox['x'] + bbox['width'], bbox['y'] + bbox['height']],
            [bbox['x'] + bbox['width'], bbox['y']],
            [bbox['x'], bbox['y'] + bbox['height']],
        ]

        edges = np.squeeze(bbox['approx'])
        for el in dst_pts:
            edges = sorted(edges, key=lambda v: sqrt((el[0] - v[0]) ** 2 + (el[1] - v[1]) ** 2))
            src_pts.append(edges[0])
            edges.pop(0)
        dst_pts = np.array(dst_pts, dtype=np.int32)
        src_pts = np.array(src_pts, dtype=np.int32)

        transform_matrix, _ = cv2.findHomography(src_pts, dst_pts)
        warp = cv2.warpPerspective(copy, transform_matrix, dsize=(copy.shape[1], copy.shape[0]))
        cropped = crop(warp, bbox)

        out[bbox['y']:bbox['y'] + bbox['height'], bbox['x']:bbox['x'] + bbox['width']] = cropped
        bbox.pop('approx', None)
        bbox["rect"] = cropped
    return out, bboxes


if __name__ == "__main__":
    bboxs = {}
    for img_i in range(1, 28):
        original = cv2.imread("../msf_lillo/{0:0=2d}.jpg".format(img_i),
                              cv2.IMREAD_UNCHANGED)
        np.swapaxes(original, 0, 1)
        bbox_list, bbox_img = run_frame(original)
        out, bbox_list = rect(original, bbox_list)
        cv2.imwrite("./output/{0:0=2d}.jpg".format(img_i), out)
        bboxs["{0:0=2d}.jpg".format(img_i)] = bbox_list

    with open("rect_bboxs.json", "w") as out:
        print(bboxs)
        json_obj = json.dumps(bboxs)
        out.write(json_obj)