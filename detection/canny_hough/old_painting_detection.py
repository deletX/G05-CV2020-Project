import cv2
import numpy as np


def get_contours(frame):
    img_cnts = frame.copy()
    img_blur = cv2.GaussianBlur(frame, (7, 7), 1)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, 23, 20)
    kernel = np.ones((5, 5))
    img_dil = cv2.dilate(img_canny, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bbox_list = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_min = 6000
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        if area > area_min and area > 0.5 * (w * h):
            print(len(approx))
            if len(approx) == 4:
                bbox = {'x': x, 'y': y, 'height': h, 'width': w, 'approx': approx}
                bbox_list.append(bbox)
                cv2.rectangle(img_cnts, (x, y), (x + w, y + h), (0, 255, 0), 5)
                cv2.putText(img_cnts, "Area: " + str(int(area)), (x + w + 20, y + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
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
