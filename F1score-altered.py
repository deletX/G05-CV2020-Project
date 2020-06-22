import json

import cv2

from detection.iou import __iou__
from detection.threshold_ccl.threshold_ccl import run_frame
from rectification.rectification import rect
from retrieval.retrieval import retrieval, load_json_file_from_path


def setup():
    query_images = load_json_file_from_path("./paintings_descriptors.json")
    dic = {}
    for i in range(1, 21):
        print("{0:0=2d}.jpg".format(i))
        original = cv2.imread("../msf_lillo/{0:0=2d}.jpg".format(i), cv2.IMREAD_UNCHANGED)
        bboxs, _ = run_frame(original)
        _, bboxs = rect(original, bboxs)
        for bbox in bboxs:
            result = retrieval(bbox["rect"], query_images)
            bbox.pop("rect", None)
            bbox["results"] = result
        dic["{0:0=2d}.jpg".format(i)] = bboxs
    return dic


def main():
    dic = setup()
    TP = 0
    FP = 0
    FN = 0
    ground_truth = load_json_file_from_path("../msf_lillo/bbox_painting.json")

    for key in ground_truth:
        for item in ground_truth[key]:
            item["detected"] = False

    for i in range(1, 21):
        print("{0:0=2d}.jpg".format(i))
        bbox_list = dic["{0:0=2d}.jpg".format(i)]
        ground_truth_list = ground_truth["{0:0=2d}.jpg".format(i)]
        for bbox in bbox_list:
            ground_truth_list.sort(key=lambda val: abs(bbox['x'] - val["x"]) + abs(bbox["y"] - val["y"]))
            if len(ground_truth_list) > 0:
                gt_bbox = ground_truth_list[0]
                if __iou__(gt_bbox, bbox) > 0.5:
                    if gt_bbox["img"] == "" or gt_bbox["img"] == bbox["results"][0][3]:
                        TP += 1
                    else:
                        FP += 1
                else:
                    FP += 1

    for key in ground_truth:
        for item in ground_truth[key]:
            if not item["detected"]:
                FN += 1

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * (p * r) / (p + r)
    print("Precision: {} \nRecall: {}\nF1-score: {}".format(p, r, F1))


if __name__ == "__main__":
    main()
