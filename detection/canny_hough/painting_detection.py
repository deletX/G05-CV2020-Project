import cv2
import numpy as np
from preprocessing.avg_histogram.avg_histogram import hist_distance


def calc_hist(frame):
    bgr_planes = cv2.split(frame)
    histSize = 256
    histRange = (0, 256)  # the upper boundary is exclusive
    hist_b = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange)
    hist_g = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange)
    hist_r = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange)
    cv2.normalize(hist_b, hist_b, 1, 0, norm_type=cv2.NORM_L2)
    cv2.normalize(hist_g, hist_g, 1, 0, norm_type=cv2.NORM_L2)
    cv2.normalize(hist_r, hist_r, 1, 0, norm_type=cv2.NORM_L2)
    return hist_b, hist_g, hist_r


def get_contours(frame):
    img_cnts = frame.copy()
    lower = np.array([0, 0, 0])
    upper = np.array([140, 130, 220])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = 255 - mask
    kernel = np.ones((5, 5))
    img_dil = cv2.dilate(mask, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bbox_list = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        hist_b, hist_g, hist_r = calc_hist(frame[y:y + h, x:x + w, :])
        if ((img_dil.shape[0] - 1) * (img_dil.shape[1] - 1) * .99) > w * h > (
                (img_dil.shape[0] - 1) * (img_dil.shape[1] - 1) * 0.005) and len(approx) == 4 and \
                hist_distance(hist_b, hist_g, hist_r, method=cv2.HISTCMP_CORREL) > 0.4:
            bbox = {'x': x, 'y': y, 'height': h, 'width': w, 'approx': approx}
            bbox_list.append(bbox)
            cv2.rectangle(img_cnts, (x, y), (x + w, y + h), (255, 0, 0), 10)
            cv2.putText(img_cnts, "Area: " + str(int(area)), (x + w + 20, y + 20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

    return bbox_list, img_cnts


def main():
    cap = cv2.VideoCapture('./VIRB0401.MP4')
    # bboxes = []
    if not cap.isOpened():
        print('Cannot open camera')
    cap.set(3, 600)
    cap.set(4, 480)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bbox_list, img_cnts = get_contours(frame)
        # bboxes.append(bbox_list)
        cv2.imshow("Result", img_cnts)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
