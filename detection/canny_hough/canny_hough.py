import numpy as np
import cv2
import glob
import preprocessing.camera_calibration.camera_calibration as my_cc
from matplotlib import pyplot as plt


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 100, 200, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares


for name in glob.glob("./input/*.jpg"):
    img = cv2.imread(name, 1)
    backtorgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #plt.imshow(backtorgb)
    #plt.show()
    #CANNY
    # edges = cv2.Canny(img,100,200)
    # plt.subplot(121),plt.imshow(backtorgb)
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()


    # HOUGH
    squares = find_squares(img)

    plt.subplot(121), plt.imshow(backtorgb)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    cv2.drawContours(backtorgb, squares, -1, (50, 168, 82), 3)

    plt.subplot(122), plt.imshow(backtorgb)
    plt.title('Squared Image'), plt.xticks([]), plt.yticks([])




    plt.show()

    #cv2.imwrite("./output/" + name, backtorgb)


#
# for name in glob.glob("./input/*.jpg"):
#     img = cv2.imread(name)
#
#     img = cv2.GaussianBlur(img, (5, 5), 0)
#
#     #img = my_cc.cc(name)
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     canny = cv2.Canny(gray, 100, 200, 1)
#
#     cnts = find_squares(img)
#         #cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     #cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#
#     for c in cnts:
#         cv2.drawContours(img, [c], 0, (0, 255, 0), 3)
#     # HOUGH squares = find_squares(img)
#     #
#     #     # plt.subplot(121), plt.imshow(backtorgb)
#     #     plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#     #
#     #     cv2.drawContours(backtorgb, squares, -1, (50, 168, 82), 3)
#     #
#     #     plt.subplot(122), plt.imshow(backtorgb)
#     #     plt.title('Squared Image'), plt.xticks([]), plt.yticks([])
#     #     plt.show()
#
#     cv2.imshow("result", img)
#     cv2.waitKey(0)
#
#     # cv2.imwrite("./output/" + name, backtorgb)
