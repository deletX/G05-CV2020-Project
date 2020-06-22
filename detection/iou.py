from statistics import mean
import copy
from detection.canny_hough.painting_detection import get_contours as p_d_bbox
from detection.threshold_ccl.threshold_ccl import run_frame as t_ccl_bbox
import cv2
import json


def setup():
    """
    Crea le bboxes come indicato dalle funzioni fornite (v. import)
    """
    p_d_bboxes = {}
    t_ccl_bboxes = {}
    for i in range(1, 28):
        t_ccl_frame = cv2.imread("./threshold_ccl/input/{0:0=2d}.jpg".format(i), cv2.IMREAD_UNCHANGED)
        p_d_frame = cv2.imread("./canny_hough/input/{0:0=2d}.jpg".format(i),
                               cv2.IMREAD_UNCHANGED)

        p_d_bbox_list, pd_img = p_d_bbox(p_d_frame)

        t_ccl_bbox_list, t_ccl_img = t_ccl_bbox(t_ccl_frame)

        cv2.imwrite("./threshold_ccl/output/{0:0=2d}.jpg".format(i), t_ccl_img)
        cv2.imwrite("./canny_hough/output/input/{0:0=2d}.jpg".format(i), pd_img)

        p_d_bboxes["{0:0=2d}.jpg".format(i)] = p_d_bbox_list
        t_ccl_bboxes["{0:0=2d}.jpg".format(i)] = t_ccl_bbox_list

    return p_d_bboxes, t_ccl_bboxes


# def __iou__(gt_bbox, ev_bbox):
#     """
#     Date una bbox ground truth ed una bbox da valutare calcola le aree, l'area di intersezione ed unione e ne ritorna il rapporto.
#     """
#     gt_top_left = (gt_bbox["x"], gt_bbox["y"])
#     gt_bot_right = (gt_bbox["x"] + gt_bbox["width"], gt_bbox["y"] + gt_bbox["height"])
#     gt_area = gt_bbox["width"] * gt_bbox["height"]
#
#     ev_top_left = (ev_bbox["x"], ev_bbox["y"])
#     ev_bot_right = (ev_bbox["x"] + ev_bbox["width"], ev_bbox["y"] + ev_bbox["height"])
#     ev_area = ev_bbox["width"] * ev_bbox["height"]
#
#     inter_rect_top_left = (max(gt_top_left[0], ev_top_left[0]), max(gt_top_left[1], ev_top_left[1]))
#     inter_rect_bot_right = (min(gt_bot_right[0], ev_bot_right[0]), min(gt_bot_right[1], ev_bot_right[1]))
#
#     intersection = abs(inter_rect_bot_right[0] - inter_rect_top_left[0]) * abs(
#         inter_rect_bot_right[1] - inter_rect_top_left[1])
#     union = gt_area + ev_area - intersection
#
#     return intersection / union

def __iou__(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
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

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
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
    Calcola la media delle IoU in base alle bbox trovate rispetto alle "Ground truth" definite nel file msf_lillo\\bbox_painting.json

    consider_undetected_paintings permette di aggiungere uno 0 per ogni dipindo non identificato

    Notes:
        Questo algoritmo cerca la bbox nel gt più vicina (manhattan distance) e poi ne calcola l'IoU.
    """
    if verbose:
        print("Running setup")
    pd_bboxes, tccl_bboxes = setup()
    pd_ious = []
    tccl_ious = []

    if verbose: print("Loading ground truth")
    with open('./../msf_lillo/bbox_painting.json') as json_file:
        ground_truth = json.load(json_file)

    for key in ground_truth:
        for item in ground_truth[key]:
            item["detected"] = False

    gt_pd = ground_truth
    gt_tccl = copy.deepcopy(ground_truth)

    pd_TP = 0
    pd_FP = 0
    pd_FN = 0
    tccl_TP = 0
    tccl_FP = 0
    tccl_FN = 0

    if verbose:
        print("Starting")
    for i in range(1, 28):
        if verbose: print("{0:0=2d}.jpg".format(i))
        gt_pd_bbox = gt_pd["{0:0=2d}.jpg".format(i)]
        gt_ttcl_bbox = gt_tccl["{0:0=2d}.jpg".format(i)]
        pd_bbox = pd_bboxes["{0:0=2d}.jpg".format(i)]
        tccl_bbox = tccl_bboxes["{0:0=2d}.jpg".format(i)]
        if verbose:
            print("checking painting_detection bbox list for {0:0=2d}.jpg".format(i))
        for bbox in pd_bbox:
            gt_pd_bbox.sort(key=lambda val: abs(bbox['x'] - val["x"]) + abs(bbox["y"] - val["y"]))
            if len(gt_pd_bbox) > 0:
                gt = gt_pd_bbox[0]
                iou = __iou__(gt, bbox)
                if (iou > true_positive_iou_threshold):
                    gt["detected"] = True
                    pd_TP += 1
                else:
                    pd_FP += 1
            else:
                iou = 0
            if verbose:
                print("Ground truth bbox: \n {} \n Bbox: \n {}\n iou: {}".format(gt, bbox, iou))
            pd_ious.append(iou)

        if verbose:
            print("checking threshold_ccl bbox list for {0:0=2d}.jpg".format(i))
        for bbox in tccl_bbox:
            gt_ttcl_bbox.sort(key=lambda val: abs(bbox["x"] - val["x"]) + abs(bbox["y"] - val["y"]))
            if len(gt_ttcl_bbox) > 0:
                gt = gt_ttcl_bbox[0]
                iou = __iou__(gt, bbox)
                if (iou > true_positive_iou_threshold):
                    gt["detected"] = True
                    tccl_TP += 1
                else:
                    tccl_FP += 1
            else:
                iou = 0
            if verbose:
                print("Ground truth bbox: \n {} \n Bbox: \n {}\n iou: {}".format(gt, bbox, iou))
            tccl_ious.append(iou)

    for key in gt_pd:
        for item in gt_pd[key]:
            if item["detected"] == False:
                pd_FN += 1

    for key in gt_tccl:
        for item in gt_tccl[key]:
            if item["detected"] == False:
                tccl_FN += 1

    pd_prec = pd_TP / (pd_TP + pd_FP)
    pd_rec = pd_TP / (pd_TP + pd_FN)
    pd_F1 = 2 * (pd_prec * pd_rec) / (pd_prec + pd_rec)

    tccl_prec = tccl_TP / (tccl_TP + tccl_FP)
    tccl_rec = tccl_TP / (tccl_TP + tccl_FN)
    tccl_F1 = 2 * (tccl_prec * tccl_rec) / (tccl_prec + tccl_rec)

    mean_pd = mean(pd_ious)
    mean_tccl = mean(tccl_ious)
    if verbose:
        print(
            "Mean painting_detection IoU: {} \nMean threshold_ccl IoU: {}\nPAINTING_DETECTION precision {}; recall {}; F1: {}.\nTHRESHOLD_CCL precision {}; recall {}; F1: {}".format(
                mean_pd, mean_tccl, pd_prec, pd_rec, pd_F1, tccl_prec, tccl_rec, tccl_F1))
    return mean(pd_ious), mean(tccl_ious)


def main():
    print(calc_iou(verbose=True))


if __name__ == "__main__":
    main()
