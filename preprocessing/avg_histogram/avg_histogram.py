from __future__ import print_function
from __future__ import division

import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

from paths import PROJ_ROOT, IMAGE_DB


def calc_hist(frame):
    """
    Given a frame compute the normalized blue, green and red histogram

    :param frame: Frame on which to compute the histograms
    :return: Normalized blue, green, red histograms
    :rtype: tuple
    """
    bgr_planes = cv2.split(frame)
    histSize = 256
    histRange = (0, 256)
    hist_b = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange)
    hist_g = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange)
    hist_r = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange)
    cv2.normalize(hist_b, hist_b, 1, 0, norm_type=cv2.NORM_L2)
    cv2.normalize(hist_g, hist_g, 1, 0, norm_type=cv2.NORM_L2)
    cv2.normalize(hist_r, hist_r, 1, 0, norm_type=cv2.NORM_L2)
    return hist_b, hist_g, hist_r


def hist_distance(hist_b, hist_g, hist_r, method=cv2.HISTCMP_KL_DIV,
                  gt_b=os.path.join(PROJ_ROOT, "preprocessing", "avg_histogram", "hist_b.npy"),
                  gt_g=os.path.join(PROJ_ROOT, "preprocessing", "avg_histogram", "hist_g.npy"),
                  gt_r=os.path.join(PROJ_ROOT, "preprocessing", "avg_histogram", "hist_r.npy")):
    """
    Computes the average distance using the given method (default Kullback–Leibler divergence)
    of the three color histograms and the locally computed ones

    :param hist_b: blue histogram
    :param hist_g: green histogram
    :param hist_r: red histogram
    :param method: distance method
    :param gt_b: path to the locally stored ground truth blue histogram
    :param gt_g: path to the locally stored ground truth blue histogram
    :param gt_r: path to the locally stored ground truth blue histogram

    :return: average distance of the three histograms
    :rtype: float
    """
    # load local histograms, if they do not exists create them
    if not (os.path.exists(gt_b) and os.path.exists(gt_g) and os.path.exists(gt_r)):
        print("Average detection histogram not found, creating it now...")
        create_average_hist()

    av_b = np.load(gt_b).astype(np.float32)
    av_g = np.load(gt_g).astype(np.float32)
    av_r = np.load(gt_r).astype(np.float32)

    # compute each distance
    dist_b = cv2.compareHist(av_b, hist_b, method=method)
    dist_g = cv2.compareHist(av_g, hist_g, method=method)
    dist_r = cv2.compareHist(av_r, hist_r, method=method)

    # return the average
    return (dist_b + dist_g + dist_r) / 3


def create_average_hist():
    """
    Creates the ground truth histogram and saves them locally

    :return:
    """

    # initialize histograms
    img_b = np.zeros((256, 1), dtype=np.float32)
    img_g = np.zeros((256, 1), dtype=np.float32)
    img_r = np.zeros((256, 1), dtype=np.float32)

    # loop over each given painting picture in the databse
    for i in range(0, 95):
        original = cv2.imread(os.path.join(IMAGE_DB, "{0:0=3d}.png".format(i)), cv2.IMREAD_UNCHANGED)

        # split bgr_planes
        bgr_planes = cv2.split(original)

        # define size and range of histograms
        histSize = 256
        histRange = (0, 256)

        # compute blue, green and red histogram
        hist_b = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange)
        hist_g = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange)
        hist_r = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange)

        img_b += hist_b
        img_g += hist_g
        img_r += hist_r

    # normalize all histograms
    cv2.normalize(img_b, img_b, 1, 0, norm_type=cv2.NORM_L2)
    cv2.normalize(img_g, img_g, 1, 0, norm_type=cv2.NORM_L2)
    cv2.normalize(img_r, img_r, 1, 0, norm_type=cv2.NORM_L2)

    # plot histogram and show
    plt.subplot(221), plt.plot(img_b)
    plt.subplot(222), plt.plot(img_g)
    plt.subplot(223), plt.plot(img_r)
    plt.savefig(os.path.join(PROJ_ROOT, "preprocessing", "avg_histogram", "bgr_avg.png"))

    # save histograms
    np.save(os.path.join(PROJ_ROOT, "preprocessing", "avg_histogram", "hist_b.npy"), img_b)
    np.save(os.path.join(PROJ_ROOT, "preprocessing", "avg_histogram", "hist_g.npy"), img_g)
    np.save(os.path.join(PROJ_ROOT, "preprocessing", "avg_histogram", "hist_r.npy"), img_r)


if __name__ == "__main__":
    create_average_hist()
