import cv2
import numpy as np
import json

from detection.canny_hough.painting_detection import get_contours


def crop(img, box):
    x_left = box['y']
    x_right = x_left + box['height']
    y_left = box['x']
    y_right = y_left + box['width']

    return img[x_left:x_right, y_left:y_right, :]


def rect(frame, bboxes):
    pass


if __name__ == "__main__":
    for img_i in range(1, 28):
        original = cv2.imread("../detection/canny_hough/input/{0:0=2d}.jpg".format(img_i),
                              cv2.IMREAD_UNCHANGED)
        np.swapaxes(original, 0, 1)
        bbox_list, bbox_img = get_contours(original)
        cv2.imshow("or", cv2.resize(original, (1920, 1080)))
        cv2.waitKey()
        out = original.copy()
        for bbox in bbox_list:
            copy = original.copy()

            dst_pts = np.array(
                [(bbox['x'], bbox['y']),
                 (bbox['x'], bbox['y'] + bbox['height']),
                 (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                 (bbox['x'] + bbox['width'], bbox['y'])],
                dtype=np.float32)
            dst_pts = dst_pts[:, np.newaxis, :]

            transform_matrix, _ = np.asarray(cv2.findHomography(bbox['approx'], dst_pts))
            warp = cv2.warpPerspective(copy, transform_matrix, dsize=(copy.shape[1], copy.shape[0]))
            cropped = crop(warp, bbox)
            out[bbox['y']:bbox['y'] + bbox['height'], bbox['x']:bbox['x'] + bbox['width']] = cropped

        cv2.imshow("or", cv2.resize(out, (1920, 1080)))
        cv2.waitKey()
