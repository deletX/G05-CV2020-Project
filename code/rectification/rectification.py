import os

import cv2
import numpy as np
from math import sqrt
from code.detection.painting.painting_detection import run_frame
from code.paths import MSF_DATASET


def crop(img, box):
    """
    Given a bounding box and a frame returns the crop corresponding to the bounding box of the frame

    :param img: frame
    :param box: bounding box
    :type box: dict
    :return: Crop
    """
    x_left = box['y']
    x_right = x_left + box['height']
    y_left = box['x']
    y_right = y_left + box['width']

    return img[x_left:x_right, y_left:y_right, :]


def rect(frame, bboxes):
    """
    Computes the rectification for each given bounding box containing the polygonal approximation of the content.
    The painting gets rectified by warping its vertices onto the bounding box ones

    :param frame: frame
    :param bboxes: Bounding boxes list
    :type bboxes: list
    :return: The image with the rectification printed on it and the list of bboxes on which
            the rectified crop had been added
    """
    out = frame.copy()
    for bbox in bboxes:
        copy = frame.copy()
        src_pts = []

        # the destination points are the bounding box vertices
        dst_pts = [
            [bbox['x'], bbox['y']],
            [bbox['x'] + bbox['width'], bbox['y'] + bbox['height']],
            [bbox['x'] + bbox['width'], bbox['y']],
            [bbox['x'], bbox['y'] + bbox['height']],
        ]

        edges = np.squeeze(bbox['approx'])
        for el in dst_pts:
            # sort the painting vertices from the nearest to the furthest from the given bounding box vertex
            edges = sorted(edges, key=lambda v: sqrt((el[0] - v[0]) ** 2 + (el[1] - v[1]) ** 2))
            src_pts.append(edges[0])

            # remove the point so it cannot be chosen again
            edges.pop(0)
        dst_pts = np.array(dst_pts, dtype=np.int32)
        src_pts = np.array(src_pts, dtype=np.int32)

        # find the Homography transformation
        transform_matrix, _ = cv2.findHomography(src_pts, dst_pts)

        # warp the copy with the found transformation matrix
        warp = cv2.warpPerspective(copy, transform_matrix, dsize=(copy.shape[1], copy.shape[0]))

        # crop the result, so to keep just the rectified painting
        cropped = crop(warp, bbox)

        # print the rectified painting
        out[bbox['y']:bbox['y'] + bbox['height'], bbox['x']:bbox['x'] + bbox['width']] = cropped

        # remove the poly from the dictionary and add the rectified painting
        bbox.pop('approx', None)
        bbox["rect"] = cropped
    return out, bboxes


def main():
    """
    Main function used for testing purposes.

    :return:
    """
    for img_i in range(1, 28):
        original = cv2.imread(os.path.join(MSF_DATASET, "{0:0=2d}.jpg".format(i)), cv2.IMREAD_UNCHANGED)
        bbox_list, bbox_img = run_frame(original)
        out, bbox_list = rect(original, bbox_list)
        cv2.imshow("result", out)
        cv2.waitKey()

        if __name__ == "__main__":
            main()
