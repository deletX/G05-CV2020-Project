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

        p_d_bbox_list, _ = p_d_bbox(p_d_frame)

        t_ccl_bbox_list, _ = t_ccl_bbox(t_ccl_frame)

        p_d_bboxes["{0:0=2d}.jpg".format(i)] = p_d_bbox_list
        t_ccl_bboxes["{0:0=2d}.jpg".format(i)] = t_ccl_bbox_list

    return p_d_bboxes,t_ccl_bboxes


def __iou__(gt_bbox, ev_bbox):
    """
    Date una bbox ground truth ed una bbox da valutare calcola le aree, l'area di intersezione ed unione e ne ritorna il rapporto.
    """
    gt_top_left = (gt_bbox["x"],gt_bbox["y"])
    gt_bot_right = (gt_bbox["x"]+gt_bbox["width"], gt_bbox["y"]+gt_bbox["height"])
    gt_area = gt_bbox["width"]*gt_bbox["height"]

    ev_top_left = (ev_bbox["x"],ev_bbox["y"])
    ev_bot_right = (ev_bbox["x"]+ev_bbox["width"], ev_bbox["y"]+ev_bbox["height"])
    ev_area = gt_bbox["width"]*gt_bbox["height"]

    inter_rect_top_left = (max(gt_top_left[0],ev_top_left[0]),max(gt_top_left[1],ev_top_left[1]))
    inter_rect_bot_right = (min(gt_bot_right[0],ev_bot_right[0]),min(gt_bot_right[1],ev_bot_right[1]))

    intersection = abs(inter_rect_bot_right[0] - inter_rect_top_left[0])*abs(inter_rect_bot_right[1] - inter_rect_top_left[1])
    union = gt_area + ev_area - intersection

    return intersection/union


def calc_iou(consider_undetected_paintings=True, verbose=False):
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
            item["detected"]=False
   
    gt_pd = ground_truth
    gt_tccl = copy.deepcopy(ground_truth)

    if verbose:
        print("Starting")
    for i in range(1, 21):
        if verbose: print("{0:0=2d}.jpg".format(i))
        gt_pd_bbox = gt_pd["{0:0=2d}.jpg".format(i)]
        gt_ttcl_bbox = gt_tccl["{0:0=2d}.jpg".format(i)]
        pd_bbox = pd_bboxes["{0:0=2d}.jpg".format(i)]
        tccl_bbox = tccl_bboxes["{0:0=2d}.jpg".format(i)]
        if verbose:
            print("checking painting_detection bbox list for {0:0=2d}.jpg".format(i))
        for bbox in pd_bbox:
            gt_pd_bbox.sort(key=lambda val: abs(bbox['x']-val["x"])+abs(bbox["y"]-val["y"]))
            if len(gt_pd_bbox) > 0:
                gt = gt_pd_bbox[0]
                iou = __iou__(gt, bbox)
                gt["detected"] = True
            else:
                iou = 0
            if verbose:
                print("Ground truth bbox: \n {} \n Bbox: \n {}\n iou: {}".format(gt, bbox, iou))
            pd_ious.append(iou)

        if verbose:
            print("checking threshold_ccl bbox list for {0:0=2d}.jpg".format(i))
        for bbox in tccl_bbox:
            gt_ttcl_bbox.sort(key=lambda val: abs(bbox["x"]-val["x"])+abs(bbox["y"]-val["y"]))
            if len(gt_ttcl_bbox) > 0:
                gt = gt_ttcl_bbox[0]
                iou = __iou__(gt, bbox)
                gt["detected"] = True
            else:
                iou=0
            if verbose:
                print("Ground truth bbox: \n {} \n Bbox: \n {}\n iou: {}".format(gt, bbox, iou))
            tccl_ious.append(iou)      

    if consider_undetected_paintings:
        for key in gt_pd:
            for item in gt_pd[key]:
                if item["detected"] == False:
                    pd_ious.append(0)
        for key in gt_tccl:
            for item in gt_tccl[key]:
                if item["detected"] == False:
                    tccl_ious.append(0)
    
    mean_pd = mean(pd_ious)
    mean_tccl = mean(tccl_ious)
    if verbose:
        print("Mean painting_detection IoU: {} \nMean threshold_ccl IoU: {}".format(mean_pd, mean_tccl))
    return mean(pd_ious), mean(tccl_ious)   


def main():
    print(calc_iou(consider_undetected_paintings=True, verbose=True))


if __name__ == "__main__":
    main()