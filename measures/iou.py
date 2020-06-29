import os
from statistics import mean
from detection.painting.painting_detection import run_frame
import cv2
import json

from paths import MSF_DATASET


def setup():
    """
    Run detection on msf_dataset frames and stores data into a dictionary

    :return: Dictionary that contains the list of bounding boxes for each frame
    :rtype dict:
    """
    bboxes = {}
    for i in range(1, 78):
        # open the frame
        frame = cv2.imread(os.path.join(MSF_DATASET, "{0:0=2d}.jpg".format(i)),
                           cv2.IMREAD_UNCHANGED)

        # run the detection algorithm
        bbox_list, img = run_frame(frame)

        # if decommented it is saved the result into ./output which must be created
        # cv2.imwrite("./output/input/{0:0=2d}.jpg".format(i), img)

        # store the bounding boxes list into the overall dictionary
        bboxes["{0:0=2d}.jpg".format(i)] = bbox_list

    return bboxes


def __iou__(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    :param bb1: Keys: {'x', 'y', 'width', 'height'}
                The (x, y) position is at the top left corner,
                the (x + width, y + height) position is at the bottom right corner

    :param bb2: Keys: {'x', 'y', 'width', 'height'}
                The (x, y) position is at the top left corner,
                the (x + width, y + height) position is at the bottom right corner
    :type bb1: dict
    :type bb2: dict

    :return: Intersection over Union in [0,1]
    :rtype: float
    """
    # assert values
    assert bb1['x'] < bb1['x'] + bb1['width']
    assert bb1['y'] < bb1['y'] + bb1['height']
    assert bb2['x'] < bb2['x'] + bb2['width']
    assert bb2['y'] < bb2['y'] + bb2['height']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x'], bb2['x'])
    y_top = max(bb1['y'], bb2['y'])
    x_right = min(bb1['x'] + bb1['width'], bb2['x'] + bb2['width'])
    y_bottom = min(bb1['y'] + bb1['height'], bb2['y'] + bb2['height'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # the intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both bounding boxes
    bb1_area = (bb1['x'] + bb1['width'] - bb1['x']) * (bb1['y'] + bb1['height'] - bb1['y'])
    bb2_area = (bb2['x'] + bb2['width'] - bb2['x']) * (bb2['y'] + bb2['height'] - bb2['y'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def calc_iou(true_positive_iou_threshold=0.5, verbose=False):
    """
    Computes the average IoU of detected bounding boxes with respect to those defined into the ground truth in bbox_painting.json
    It also computes the F1 score considering a TP if the iou value is above a defined threshold.

    :param true_positive_iou_threshold: the threshold at which above is considered a TP, FP otherwise. Default 0.5
    :param verbose: enable verbose mode. Default: False
    :type true_positive_iou_threshold: float
    :type verbose: bool
    :return: average IoU, Precision, Recall and F1 score
    :rtype: (float, float, float, float)
    """
    if verbose: print("Running setup")
    bboxes = setup()
    ious = []

    if verbose: print("Loading ground truth")
    with open(os.path.join(MSF_DATASET, "bbox_painting.json")) as json_file:
        ground_truth = json.load(json_file)

    # define detected, to detect FN
    for key in ground_truth:
        for item in ground_truth[key]:
            item["detected"] = False

    # define True Positives (TP), False Positives (FP), False Negatives (FN)
    TP = 0
    FP = 0
    FN = 0

    if verbose: print("Starting")

    for i in range(1, 78):
        if verbose: print("{0:0=2d}.jpg".format(i))

        # get the ground truth bounding and detected bounding box list for the frame
        gt_bbox_list = ground_truth["{0:0=2d}.jpg".format(i)]
        bbox_list = bboxes["{0:0=2d}.jpg".format(i)]

        for bbox in bbox_list:

            # sort the ground truth list with respect to the current bounding box distance
            gt_bbox_list.sort(key=lambda val: abs(bbox['x'] - val["x"]) + abs(bbox["y"] - val["y"]))

            if len(gt_bbox_list) > 0:

                # the nearest ground truth bounding box is compared with the current bounding box
                gt = gt_bbox_list[0]
                iou = __iou__(gt, bbox)
            else:
                # if there is no ground truth bounding box for the current frame the iou is set to 0
                iou = 0
            if iou > true_positive_iou_threshold:
                gt["detected"] = True
                TP += 1
            else:
                FP += 1

            ious.append(iou)

    # compute the false negatives that have not been detected
    for key in ground_truth:
        for item in ground_truth[key]:
            if not item["detected"]:
                FN += 1

    # compute precision, recall and F1-score
    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
    F1 = 2 * (prec * rec) / (prec + rec)

    # compute the mean of the various compute IoUs
    mean_iou = mean(ious)
    if verbose: print("Mean IoU: {}\n Precision {}; Recall {}; F1: {}".format(
        mean_iou, prec, rec, F1))
    return mean_iou


def main():
    print(calc_iou(verbose=True))


if __name__ == "__main__":
    main()
