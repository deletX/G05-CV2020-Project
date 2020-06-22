from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
from matplotlib import pyplot as plt


def hist_distance(hist_b, hist_g, hist_r, method=cv2.HISTCMP_KL_DIV):
    av_b = np.load("./hist_b.npy").astype(np.float32)
    av_g = np.load("./hist_g.npy").astype(np.float32)
    av_r = np.load("./hist_r.npy").astype(np.float32)
    dist_b = cv2.compareHist(av_b, hist_b, method=method)
    dist_g = cv2.compareHist(av_g, hist_g, method=method)
    dist_r = cv2.compareHist(av_r, hist_r, method=method)
    return (dist_b + dist_g + dist_r) / 3


if __name__ == "__main__":
    img_b = np.zeros((256, 1), dtype=np.float32)
    img_g = np.zeros((256, 1), dtype=np.float32)
    img_r = np.zeros((256, 1), dtype=np.float32)
    for i in range(0, 95):
        original = cv2.imread("../../retrieval/paintings_db/{0:0=3d}.png".format(i),
                              cv2.IMREAD_UNCHANGED)
        bgr_planes = cv2.split(original)

        histSize = 256
        histRange = (0, 256)  # the upper boundary is exclusive

        hist_b = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange)
        hist_g = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange)
        hist_r = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange)

        img_b += hist_b
        img_g += hist_g
        img_r += hist_r
    # for i in range(1, histSize):
    #     cv2.line(histImage, (bin_w * (i - 1), hist_h - int(round(hist_b[i - 1]))),
    #              (bin_w * (i), hist_h - int(round(hist_b[i]))),
    #              (255, 0, 0), thickness=2)
    #     cv2.line(histImage, (bin_w * (i - 1), hist_h - int(round(hist_g[i - 1]))),
    #              (bin_w * (i), hist_h - int(round(hist_g[i]))),
    #              (0, 255, 0), thickness=2)
    #     cv2.line(histImage, (bin_w * (i - 1), hist_h - int(round(hist_r[i - 1]))),
    #              (bin_w * (i), hist_h - int(round(hist_r[i]))),
    #              (0, 0, 255), thickness=2)
    #
    # cv2.imshow('Source image', original)
    # cv2.imshow('calcHist Demo', histImage)
    # cv2.waitKey()
    hist_w = 512
    hist_h = 400
    bin_w = int(round(hist_w / 256))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    cv2.normalize(img_b, img_b, 1, 0, norm_type=cv2.NORM_L2)
    cv2.normalize(img_g, img_g, 1, 0, norm_type=cv2.NORM_L2)
    cv2.normalize(img_r, img_r, 1, 0, norm_type=cv2.NORM_L2)
    plt.subplot(221), plt.plot(img_b)
    plt.subplot(222), plt.plot(img_g)
    plt.subplot(223), plt.plot(img_r)
    plt.show()
    np.save("./hist_b", img_b)
    np.save("./hist_g", img_g)
    np.save("./hist_r", img_r)
