# import the necessary packages
import argparse
import os

import cv2

# import local packages
from detection.people.yolo_func import yolo_func
from detection.painting.painting_detection import run_frame
from localization.localization import localization
from rectification.rectification import rect
from retrieval.retrieval import retrieval, load_json_file_from_path
from retrieval.setup_db import create_painting_db
from paths import PROJ_ROOT, YOLO_WEIGHTS_PATH, YOLO_CFG_PATH


def yolo_setup(weights_path=YOLO_WEIGHTS_PATH, config_path=YOLO_CFG_PATH, verbose=False):
    """
    Reads the weights and config files and prepares the network for the YOLO to run.

    :param weights_path: The path for the weights file.
    :param config_path: The path for the config file.
    :type weights_path: str
    :type config_path: str
    :return: Network and layer names for yolo
    :rtype: tuple
    """
    if not os.path.exists(YOLO_WEIGHTS_PATH):
        print("YOLO weights not found, you can find the link in the readme")
        exit(-2)
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, ln


def show_and_wait(img):
    """
    Debug function. Show an image and waits for a key press

    :param img: Image to be shown
    """
    cv2.imshow("img", img)
    cv2.waitKey()


def get_params():
    """
    Verify presence of required arguments and reads passed arguments into a dictionary

    :return: Dictionary of arguments
    :rtype: dict
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="path to input video")
    ap.add_argument("-o", "--output", default=argparse.SUPPRESS,
                    help="path to the output name")
    ap.add_argument("-v", "--verbose", default=argparse.SUPPRESS, action='store_true',
                    help="verbose mode")
    ap.add_argument("-d", "--debug", default=argparse.SUPPRESS, action='store_true',
                    help="slow mode")
    ap.add_argument("-s", "--skip", default=1, type=int,
                    help="slow mode")
    return vars(ap.parse_args())


if __name__ == "__main__":
    args = get_params()

    # define if verbose mode
    verbose = "verbose" in args

    # define if debug mode
    debug = "debug" in args

    # define skip for testing purposes
    skip = args["skip"]
    if verbose: print("Analyzing 1 frame each {}".format(skip)) if skip > 1 else print("Analyzing all frames")

    # YOLO setup
    if verbose: print("Loading Yolo...")
    net, ln = yolo_setup(verbose=verbose)

    # retrieval setup, load db
    if verbose: print("Loading painting descriptors...")
    if not os.path.exists(os.path.join(PROJ_ROOT, "retrieval", "./paintings_descriptors.json")):
        if verbose: print("Descriptors not found, creating them...")
        create_painting_db()
    query_images = load_json_file_from_path(os.path.join(PROJ_ROOT, "retrieval", "paintings_descriptors.json"))

    # color definition
    painting_bbox_color = [255, 0, 0]
    painting_bbox_width = 7
    painting_descr_color = [255, 255, 255]
    painting_descr_size = .4
    painting_descr_thick = 1

    person_bbox_color = [0, 0, 255]
    person_bbox_width = 5
    person_descr_color = [255, 255, 255]
    person_descr_size = .4
    person_descr_thick = 1

    # open input
    if verbose: print("Opening input video...")
    cap = cv2.VideoCapture(args["input"])
    if not cap.isOpened():
        print('Cannot open camera :(')
        exit(-1)

    # retrieve capture parameters
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) / skip
    if verbose: print("Input details:\n - {}px\n - {}px\n - {}fps".format(width, height, fps))

    # open output if defined
    if "output" in args:
        if verbose: print("Creating output file {}...".format(args["output"]))
        out = cv2.VideoWriter(os.path.join(PROJ_ROOT, args["output"] + ".mp4v"),
                              cv2.VideoWriter_fourcc('d', 'i', 'v', 'x'), fps,
                              (width, height))

    frame_cnt = 0
    # loop over the whole video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_cnt % skip != 0:
            if verbose: print("Skipping frame #{}".format(frame_cnt))
            frame_cnt += 1
            continue
        else:
            frame_cnt += 1
        if verbose: print("Frame #{}".format(frame_cnt))

        # object detection
        if verbose: print("Detecting paintings...")
        painting_bboxs, img_cnts = run_frame(frame)
        if verbose: print("Detected {} paintings".format(len(painting_bboxs)))
        if debug: show_and_wait(img_cnts)

        # rectification
        rect_frame, painting_bboxs = rect(frame, painting_bboxs)
        if verbose: print("Rectified")
        if debug: show_and_wait(rect_frame)

        # retrieval
        if verbose: print("Retrieving paintings from DB...")
        for bbox in painting_bboxs:
            results = retrieval(bbox["rect"], query_images)

            # remove rectification from dictionary
            bbox.pop("rect", None)

            # if there is a result add it to the dictionary
            if results is not None:
                result = results[0]
                bbox["painting"] = {
                    "title": result[0],
                    "author": result[1],
                    "room": result[2]
                }
        if verbose: print("Retrieved paintings from DB")

        # people detection
        if verbose: print("Detecting people...")
        people_bboxs = yolo_func(frame.copy(), net, ln)
        if verbose: print("Detected {} people".format(len(people_bboxs)))

        # localization
        if len([bbox for bbox in painting_bboxs if "painting" in bbox]) > 0:
            room = localization(painting_bboxs)
        else:
            room = -1
        if verbose: print("Room computed: {}".format(room))

        # Drawing painting_bboxs
        if verbose: print("Drawing...")
        for bbox in painting_bboxs:
            # draw bbox
            (x, y, w, h) = (bbox["x"], bbox["y"], bbox["width"], bbox["height"])
            cv2.rectangle(rect_frame, (x, y), (x + w, y + h), painting_bbox_color, painting_bbox_width)

            # determine description
            if "painting" in bbox:
                painting = bbox["painting"]
                title = "Title: {}".format(painting["title"])
                author = "Author: {}".format(painting["author"])
            else:
                title = "Title: ?"
                author = "Author: ?"

            # draw description
            cv2.putText(rect_frame, title, (x + 20, y + 20),
                        cv2.FONT_HERSHEY_COMPLEX, painting_descr_size, painting_descr_color, painting_descr_thick)
            cv2.putText(rect_frame, author, (x + 20, y + 40),
                        cv2.FONT_HERSHEY_COMPLEX, painting_descr_size, painting_descr_color, painting_descr_thick)

        # Drawing ppl_bboxs
        for bbox in people_bboxs:
            # draw box
            (x, y, w, h) = (bbox["x"], bbox["y"], bbox["width"], bbox["height"])
            cv2.rectangle(rect_frame, (x, y), (x + w, y + h), person_bbox_color, person_bbox_width)

            # detetrmine description (in this case the identified room)
            descr = "Room: {}".format(room if room > 0 else "?")

            # draw description
            cv2.putText(rect_frame, descr, (x + person_descr_thick * 5, y + (person_descr_thick * 5)),
                        cv2.FONT_HERSHEY_COMPLEX, person_descr_size, person_descr_color, person_descr_thick)

        if verbose: print("Exporting result...")
        # write resulting frame
        if "output" in args:
            out.write(rect_frame)
        else:
            cv2.imshow("result", rect_frame)
            cv2.waitKey(10)

    # destroy and release
    if verbose: print("Releasing resources and exiting, this may take a while!")
    cv2.destroyAllWindows()
    cap.release()
    if "output" in args:
        out.release()
    exit(0)
