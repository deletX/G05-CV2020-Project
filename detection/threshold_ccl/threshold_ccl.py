import glob

import cv2
import numpy as np
import json
from detection.canny_hough.painting_detection import calc_hist
from preprocessing.avg_histogram.avg_histogram import hist_distance


def run_frame(frame):
    """
    Run the painting detection using image processing tools.

    This technique uses OTSU thresholding and Satoshi Suzuki et al. algorithm

    :param frame: frame to be analyzed
    :return: The list of the detected bounding boxes and the frame with drawn the bounding boxes.
    :rtype: tuple
    """
    # initialize the bounding box list img_list
    img_list = []
    out = frame.copy()

    # convert the frame into HSV and split the channels, we keep just the value
    _, _, img = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))

    # apply a gaussian blur
    img = cv2.GaussianBlur(v, (7, 7), 1)

    # apply a treshold using the OTSU method
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply 1 dilation and 5 opening
    kernel = np.ones((3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=5)

    # flip the detection (we were detecting the background).
    # It is technically useless, but for debugging purposes it is best to show "detected" paintings
    img = 255 - img

    # use Satoshi Suzuki et al. algorithm to find the paintings contours
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over all the detected contours
    for cnt in contours:

        # compute the conour area and perimter
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)

        # approximate the contour with a polygonal curve with a precision which is a function of the perimeter length
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)

        # define the bounding box of the contour
        x, y, w, h = cv2.boundingRect(approx)

        # we compute the histogram of color in the bounding box area
        hist_b, hist_g, hist_r = calc_hist(frame[y:y + h, x:x + w, :])

        # we apply some filters to filter out:
        # - the too small and too large bounding boxes,
        # - the bounding boxes that have an histogram of colors too distant from the computed average
        # - the bounding boxes that enclose a curve that covers less than 50% of the box area
        # - the bounding boxes of curves that have not 4 sides (4 vertices)
        if ((img.shape[0] - 1) * (img.shape[1] - 1) * .99) > w * h > (
                (img.shape[0] - 1) * (img.shape[1] - 1) * 0.005) and \
                hist_distance(hist_b, hist_g, hist_r, method=cv2.HISTCMP_INTERSECT) > 3 and \
                area > 5000 and area > 0.5 * (w * h) and len(approx) == 4:
            bbox = {'x': x, 'y': y, 'height': h, 'width': w, 'approx': approx}
            img_list.append(bbox)
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 10)
            cv2.putText(out, "w/h: " + str(w / h), (x + w + 20, y + 20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

    return img_list, out


def main():
    """
    Testing main function for qualitative evaluation onto a set video
    """
    # open the video
    cap = cv2.VideoCapture('./VIRB0401.mp4')
    if not cap.isOpened():
        print('Cannot open camera')

    # loop over the video
    while True:
        # retrieve the frame
        ret, frame = cap.read()
        if not ret:
            break

        # run the detection
        _, img = run_frame(frame)

        # resize and show the result
        img = cv2.resize(img, (600, 600))
        cv2.imshow("Result", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main_check():
    """
    Run the detection onto each frame of each video to compute the average painting detection per frame for each given video.
    This has been used as a very qualitative measure
    
    :return:
    """
    res = {}

    # loop over each video of each videos subfolder
    for filename in glob.glob("D:\\Users\\gavio\\CV30LperNick\\project_material\\videos\\*\\*"):
        print(filename)

        # open the video
        capture = cv2.VideoCapture(filename)

        sum = 0
        count = 0

        # loop over each frame
        while True:
            ret, frame = capture.read()
            if frame is None:
                break

            # run detection
            list, _ = run_frame(frame)

            # increment the detection sum and frame count
            sum += len(list)
            count += 1
        # compute the average detection per frame
        avg = sum / count

        # add the result into a dictionary
        res[filename.split("\\")[-1]] = avg
        capture.release()
        cv2.destroyAllWindows()

    # all computed averages are saved into a .json file
    with open("res.json", "w") as out:
        json_obj = json.dumps(res)
        out.write(json_obj)


if __name__ == "__main__":
    main()
