import cv2
import numpy as np


def empty(a):
    pass


def getContours(img, imgContour):
    bbox_list = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = 10000
        if area > areaMin:
            #cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(len(approx))
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                bbox = {'x': x, 'y': y, 'h': h, 'w': w}
                bbox_list.append(bbox)
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
                cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20),
                            cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
                cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    return bbox_list


def main():
    cap = cv2.VideoCapture('./VIRB0401.MP4')
    if not cap.isOpened():
        print('Cannot open camera')
    cap.set(3, 600)
    cap.set(4, 480)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        imgContour = frame.copy()
        imgBlur = cv2.GaussianBlur(frame, (7, 7), 1)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        imgCanny = cv2.Canny(imgGray, 23, 20)
        kernel = np.ones((5, 5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
        bbox_list = getContours(imgDil, imgContour)
        cv2.imshow("Result", imgContour)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
