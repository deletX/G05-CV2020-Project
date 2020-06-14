import cv2
import numpy as np


def nothing(x):
    pass


def main():
    cv2.namedWindow('image')
    cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)
    cv2.setTrackbarPos('HMax', 'image', 140)
    cv2.setTrackbarPos('SMax', 'image', 130)
    cv2.setTrackbarPos('VMax', 'image', 230)
    capture = cv2.VideoCapture('./20180206_114408.mp4')
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')
        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])
        img_cnts = frame.copy()
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
            if area > 30000 and area > 0.5*(w*h) and len(approx) == 4:
                bbox = {'x': x, 'y': y, 'height': h, 'width': w, 'approx': approx}
                bbox_list.append(bbox)
                cv2.rectangle(img_cnts, (x, y), (x + w, y + h), (255, 0, 0), 10)
                cv2.putText(img_cnts, "Area: " + str(int(area)), (x + w + 20, y + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
        v = cv2.resize(img_cnts, (500, 500))
        cv2.imshow('Result', v)
        cv2.waitKey(10)


if __name__ == '__main__':
    main()