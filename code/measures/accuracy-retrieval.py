import os

import cv2
from code.measures.iou import __iou__
from code.detection.painting.painting_detection import run_frame
from code.paths import PROJ_ROOT, MSF_DATASET
from code.rectification.rectification import rect
from code.retrieval.retrieval import retrieval, load_json_file_from_path
from code.retrieval.setup_db import create_painting_db


def setup():
    """
    Setups the computation by executing the painting detection, rectification and retrieval step for each frame
    and return the dictionary with the result for each frame

    :return: Dictionary with the results up to the retrieval step
    :rtype: dict
    """

    # load the painting descriptors db
    if not os.path.exists(os.path.join(PROJ_ROOT, "retrieval", "paintings_descriptors.json")):
        print("Painting descriptors db not found, creating it now...")
        create_painting_db()

    query_images = load_json_file_from_path(os.path.join(PROJ_ROOT, "retrieval", "paintings_descriptors.json"))
    dic = {}
    for i in range(1, 78):
        print("{0:0=2d}.jpg".format(i))
        original = cv2.imread(os.path.join(MSF_DATASET, "{0:0=2d}.jpg".format(i)), cv2.IMREAD_UNCHANGED)

        # run the painting detection
        bboxs, _ = run_frame(original)

        # run the rectification
        _, bboxs = rect(original, bboxs)

        # run the retrieval step
        for bbox in bboxs:
            result = retrieval(bbox["rect"], query_images)
            bbox.pop("rect", None)
            bbox["results"] = result

        # add the results into the dictionary
        dic["{0:0=2d}.jpg".format(i)] = bboxs
    return dic


def main():
    """
    Compute the Accuracy for retrieval

    :return:
    """
    # run the detection, rectification and retreival
    dic = setup()
    trues = 0
    total = 0

    # load the ground truth
    ground_truth = load_json_file_from_path(os.path.join(MSF_DATASET, "bbox_painting.json"))

    # add a detected flag, used to compute the false negatives
    for key in ground_truth:
        for item in ground_truth[key]:
            item["detected"] = False
    # loop over the labeled frames
    for i in range(1, 78):

        bbox_list = dic["{0:0=2d}.jpg".format(i)]
        ground_truth_list = ground_truth["{0:0=2d}.jpg".format(i)]
        original = cv2.imread(os.path.join(MSF_DATASET, "{0:0=2d}.jpg".format(i)), cv2.IMREAD_UNCHANGED)

        for bbox in bbox_list:
            # the ground truth bbox is considered the closest from the detected one
            ground_truth_list.sort(key=lambda val: abs(bbox['x'] - val["x"]) + abs(bbox["y"] - val["y"]))
            if len(ground_truth_list) > 0:
                gt_bbox = ground_truth_list[0]

                cv2.rectangle(original, (gt_bbox['x'], gt_bbox['y']),
                              (gt_bbox['x'] + gt_bbox['width'], gt_bbox['y'] + gt_bbox['height']), (0, 0, 0), 10)
                cv2.rectangle(original, (bbox['x'], bbox['y']),
                              (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']), (255, 255, 255), 10)
                iou = __iou__(gt_bbox, bbox)

                if iou > 0.5 and gt_bbox["img"] != "":
                    if gt_bbox["img"] == bbox["results"][0][3]:
                        trues += 1
                    total += 1

    print("Accuracy: {}".format(trues / total)) if total != 0 else print("No values found")
    return trues / total if total != 0 else -1


if __name__ == "__main__":
    main()
