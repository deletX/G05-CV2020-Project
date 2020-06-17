# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

from detection.canny_hough.painting_detection import get_contours
from detection.people.yolo_func import yolo_func
from localization.localization import localization
from rectification.rect_pd_test import rect
from retrieval.retrieval import retrieval, load_json_file_from_path


def yolo_setup():
    weights_path = "./detection/people/yolo-coco/yolov3.weights"
    config_path = "./detection/people/yolo-coco/yolov3.cfg"
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, ln


def show_and_wait(img):
    cv2.imshow("img", img)
    cv2.waitKey()


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="path to input video")
    ap.add_argument("-o", "--output", default=argparse.SUPPRESS,
                    help="path to the output name")
    ap.add_argument("-v", "--verbose", default=argparse.SUPPRESS, action='store_true',
                    help="verbose mode")
    ap.add_argument("-s", "--slow", default=argparse.SUPPRESS, action='store_true',
                    help="slow mode")
    args = vars(ap.parse_args())

    # define verbose mode
    verbose = "verbose" in args

    # define slow mode
    slow = "slow" in args

    # YOLO setup
    if verbose: print("Loading Yolo")
    net, ln = yolo_setup()

    # retrieval setup, load db
    if verbose: print("Loading painting descriptors")
    query_images = load_json_file_from_path("./retrieval/paintings_descriptors.json")

    # color definition
    painting_bbox_color = [255, 0, 0]
    painting_bbox_width = 7
    painting_descr_color = [255, 255, 255]
    painting_descr_size = .7
    painting_descr_thick = 10

    person_bbox_color = [0, 0, 255]
    person_bbox_width = 5
    person_descr_color = [255, 255, 255]
    person_descr_size = .8
    person_descr_thick = 10

    # open input
    if verbose: print("Opening input video")
    cap = cv2.VideoCapture(args["input"])
    if not cap.isOpened():
        print('Cannot open camera')
        exit(-1)

    # retrieve capture parameters
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    if verbose: print("Input details:\n - {}px\n - {}px\n - {}fps\n - {}".format(width, height, fps, fourcc))

    # open output if defined

    if "output" in args:
        if verbose: print("Creating output file {}".format(args["output"]))
        out = cv2.VideoWriter(args["output"] + ".mp4", fourcc, fps, (width, height))

    # loop over the whole video
    while True:
        if verbose: print("Reading a frame")
        ret, frame = cap.read()
        if not ret:
            break

        # object detection

        painting_bboxs, img_cnts = get_contours(frame)
        if verbose: print("Identifying objects: {} paintings".format(len(painting_bboxs)))
        if slow: show_and_wait(img_cnts)

        # rectification
        rect_frame, _ = rect(frame, painting_bboxs)
        if verbose: print("Rectified")
        if slow: show_and_wait(rect_frame)

        # retrieval
        painting_bboxs_with_retrieval = retrieval(rect_frame, query_images, painting_bboxs)
        if verbose: print("Retrieved paintings from db")

        # people detection
        people_bboxs = yolo_func(frame.copy(), net, ln)
        if verbose: print("Detecting people: {} people found".format(len(people_bboxs)))

        # localization
        room = localization(painting_bboxs_with_retrieval)
        if verbose: print("Room computed".format(room))

        # Drawing painting_bboxs
        if verbose: print("Drawing")
        for bbox in painting_bboxs_with_retrieval:
            # draw bbox
            (x, y, w, h) = (bbox["x"], bbox["y"], bbox["width"], bbox["height"])
            cv2.rectangle(rect_frame, (x, y), (x + w, y + h), painting_bbox_color, painting_bbox_width)

            # determine description
            descr = "Title: {}\nAuthor: {}"
            if "painting" in bbox:
                painting = bbox["painting"]
                descr.format(painting["title"], painting["author"])
            else:
                descr.format("?", "?")

            # draw description
            cv2.putText(rect_frame, descr, (x + painting_descr_thick * 2, y + (painting_descr_thick * 2)),
                        cv2.FONT_HERSHEY_COMPLEX, painting_descr_size, painting_descr_color, painting_descr_thick)

        # Drawing ppl_bboxs
        for bbox in people_bboxs:
            # draw box
            (x, y, w, h) = (bbox["x"], bbox["y"], bbox["width"], bbox["height"])
            cv2.rectangle(rect_frame, (x, y), (x + w, y + h), person_bbox_color, person_bbox_width)

            # detetrmine description (in this case the identified room)
            descr = "Room: {}".format(room)

            # draw description
            cv2.putText(rect_frame, descr, (x + person_descr_thick * 2, y + (person_descr_thick * 2)),
                        cv2.FONT_HERSHEY_COMPLEX, person_descr_size, person_descr_color, person_descr_thick)

        # write resulting frame
        if "output" in args:
            out.write(rect_frame)
        else:
            cv2.imshow("result", rect_frame)
