import cv2
from detection.canny_hough.painting_detection import get_contours


def main():
    capture = cv2.VideoCapture('./20180206_113600.mp4')
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        bboxs, result = get_contours(frame)
        result = cv2.resize(result, (500, 500))
        cv2.imshow('Result', result)
        cv2.waitKey(10)


if __name__ == '__main__':
    main()