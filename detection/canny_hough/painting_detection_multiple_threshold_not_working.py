import cv2
import numpy as np


def iou(gt_bbox, ev_bbox):
    """
    Date una bbox ground truth ed una bbox da valutare calcola le aree, l'area di intersezione ed unione e ne ritorna il rapporto.
    """
    gt_top_left = (gt_bbox["x"], gt_bbox["y"])
    gt_bot_right = (gt_bbox["x"] + gt_bbox["width"], gt_bbox["y"] + gt_bbox["height"])
    gt_area = gt_bbox["width"] * gt_bbox["height"]

    ev_top_left = (ev_bbox["x"], ev_bbox["y"])
    ev_bot_right = (ev_bbox["x"] + ev_bbox["width"], ev_bbox["y"] + ev_bbox["height"])
    ev_area = ev_bbox["width"] * ev_bbox["height"]

    inter_rect_top_left = (max(gt_top_left[0], ev_top_left[0]), max(gt_top_left[1], ev_top_left[1]))
    inter_rect_bot_right = (min(gt_bot_right[0], ev_bot_right[0]), min(gt_bot_right[1], ev_bot_right[1]))

    intersection = abs(inter_rect_bot_right[0] - inter_rect_top_left[0]) * abs(
        inter_rect_bot_right[1] - inter_rect_top_left[1])
    union = gt_area + ev_area - intersection

    return intersection / union


def get_contours(frame):
    img_cnts = frame.copy()
    bbox_list = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask0 = 255 - cv2.inRange(hsv, np.array([0, 0, 0]), np.array([140, 130, 220]))
    mask1 = 255 - cv2.inRange(hsv, np.array([0, 0, 0]), np.array([100, 160, 200]))
    mask3 = 255 - cv2.inRange(hsv, np.array([18, 0, 0]), np.array([27, 196, 160]))
    mask2 = 255 - cv2.inRange(hsv, np.array([16, 0, 0]), np.array([28, 255, 222]))
    mask4 = 255 - cv2.inRange(hsv, np.array([0, 0, 0]), np.array([27, 172, 151]))
    mask5 = 255 - cv2.inRange(hsv, np.array([21, 0, 0]), np.array([45, 196, 255]))
    mask6 = 255 - cv2.inRange(hsv, np.array([0, 0, 0]), np.array([44, 65, 255]))
    masks = [mask0, mask1, mask2, mask3, mask4, mask5, mask6]
    for mask in masks:
        kernel = np.ones((5, 5))
        img_dil = cv2.dilate(mask, kernel, iterations=2)
        _, contours, hierarchy = cv2.findContours(img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox_list = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            bbox = {'x': x, 'y': y, 'height': h, 'width': w, 'approx': approx}
            if ((img_dil.shape[0] - 1) * (img_dil.shape[1] - 1) * .80) > w * h > (
                    (img_dil.shape[0] - 1) * (img_dil.shape[1] - 1) * .02) and \
                    area > 0.8 * (w * h) and \
                    len(approx) == 4:
                if len(bbox_list) > 0:
                    bbox_list.sort(key=lambda val: abs(bbox['x'] - val["x"]) + abs(bbox["y"] - val["y"]))
                    if iou(bbox_list[0], bbox) < 0.5:
                        bbox_list.append(bbox)
                        cv2.rectangle(img_cnts, (x, y), (x + w, y + h), (255, 0, 0), 10)
                        cv2.putText(img_cnts, "Area: " + str(int(area)), (x + w + 20, y + 20),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    bbox_list.append(bbox)
                    cv2.rectangle(img_cnts, (x, y), (x + w, y + h), (255, 0, 0), 10)
                    cv2.putText(img_cnts, "Area: " + str(int(area)), (x + w + 20, y + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

    if len(bbox_list) > 0:
        min_area = min(bbox_list, key=lambda val: val['width'] * val['height'])
        min_area = min_area['width'] * min_area['height']
        bbox_list = [bbox for bbox in bbox_list if (bbox['width'] * bbox['height']) < 2 * min_area]

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
