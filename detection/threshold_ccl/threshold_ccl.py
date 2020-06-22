import glob

import cv2
import numpy as np
import json
from detection.canny_hough.painting_detection import calc_hist
from preprocessing.avg_histogram.avg_histogram import hist_distance


def run_frame(frame):
    img_list = []
    out = frame.copy()
    h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))

    # clahe = cv2.createCLAHE()
    # v = clahe.apply(v)
    img = cv2.GaussianBlur(v, (7, 7), 1)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3))
    # img = cv2.medianBlur(img, 3)
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=5)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=3)
    # img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, iterations=5)
    img = 255 - img
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        hist_b, hist_g, hist_r = calc_hist(frame[y:y + h, x:x + w, :])
        if ((img.shape[0] - 1) * (img.shape[1] - 1) * .99) > w * h > (
                (img.shape[0] - 1) * (img.shape[1] - 1) * 0.005) and \
                hist_distance(hist_b, hist_g, hist_r, method=cv2.HISTCMP_INTERSECT) > 3 and \
                area > 5000 and area > 0.5 * (w * h) and len(approx) == 4:
            bbox = {'x': x, 'y': y, 'height': h, 'width': w, 'approx': approx}
            img_list.append(bbox)
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 10)
            cv2.putText(out, "Area: " + str(int(area)), (x + w + 20, y + 20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

    return img_list, out


def main():
    cap = cv2.VideoCapture('../canny_hough/VIRB0401.mp4')
    # bboxes = []
    if not cap.isOpened():
        print('Cannot open camera')
    # cap.set(3, 600)
    # cap.set(4, 480)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, img = run_frame(frame)
        img = cv2.resize(img, (600, 600))
        cv2.imshow("Result", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main_check():
    res = {}
    for filename in glob.glob("D:\\Users\\gavio\\CV30LperNick\\project_material\\videos\\*\\*"):
        print(filename)
        capture = cv2.VideoCapture(filename)
        sum = 0
        count = 0
        while True:
            ret, frame = capture.read()
            if frame is None:
                break
            list, _ = run_frame(frame)
            sum += len(list)
            count += 1
        avg = sum / count
        res[filename.split("\\")[-1]] = avg
        capture.release()
        cv2.destroyAllWindows()
    with open("res.json", "w") as out:
        json_obj = json.dumps(res)
        out.write(json_obj)


if __name__ == "__main__":
    main()
