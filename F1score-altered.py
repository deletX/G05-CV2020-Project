import cv2
from detection.iou import __iou__
from detection.threshold_ccl.threshold_ccl import run_frame
from rectification.rectification import rect
from retrieval.retrieval import retrieval, load_json_file_from_path


def setup():
    """
    Setups the computation by executing the painting detection, rectification and retrieval step for each frame
    and return the dictionary with the result for each frame

    :return: Dictionary with the results up to the retrieval step
    :rtype: dict
    """

    # load the painting descriptors db
    query_images = load_json_file_from_path("./retrieval/paintings_descriptors.json")
    dic = {}
    for i in range(1, 78):
        print("{0:0=2d}.jpg".format(i))
        original = cv2.imread("./msf_lillo/{0:0=2d}.jpg".format(i), cv2.IMREAD_UNCHANGED)

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
    Compute the F1-score on the labeled dataset

    :return:
    """
    # run the detection, rectification and retreival
    dic = setup()
    TP = 0
    FP = 0
    FN = 0

    # load the ground truth
    ground_truth = load_json_file_from_path("./msf_lillo/bbox_painting.json")

    # add a detected flag, used to compute the false negatives
    for key in ground_truth:
        for item in ground_truth[key]:
            item["detected"] = False
    # loop over the labeled frames
    for i in range(1, 78):

        bbox_list = dic["{0:0=2d}.jpg".format(i)]
        ground_truth_list = ground_truth["{0:0=2d}.jpg".format(i)]
        original = cv2.imread("./msf_lillo/{0:0=2d}.jpg".format(i), cv2.IMREAD_UNCHANGED)

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

                if iou > 0.5:

                    if gt_bbox["img"] == "" or gt_bbox["img"] == bbox["results"][0][3]:
                        # a true positive is so if it has at least .5 iou and,
                        # if it is in the db, the retrieval is correct
                        TP += 1
                        gt_bbox["detected"] = True
                    else:
                        FP += 1
                else:
                    FP += 1
    # measure the false negatives
    for key in ground_truth:
        if key in ["{0:0=2d}.jpg".format(i) for i in range(1, 78)]:
            for item in ground_truth[key]:
                if not item["detected"]:
                    FN += 1

    # compute precision recall and F1
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * (p * r) / (p + r)
    print("Precision: {} \nRecall: {}\nF1-score: {}".format(p, r, F1))


if __name__ == "__main__":
    main()
