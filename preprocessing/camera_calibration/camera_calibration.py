import numpy as np
import cv2
import glob

for name in glob.glob("./input/*.jpg"):
    img = cv2.imread(name, 1)
    intrinsic_matrix = np.array([[1230, 0., 960], [0., 1200, 540], [0., 0., 1.]]) #GoPro parameters
    distCoeff = np.array([-0.32, 0.126, 0, -0.001, -0.015]) #Gopro distorsion coefficients
    result = cv2.undistort(img, intrinsic_matrix, distCoeff, None)
    cv2.imwrite("./output/" + name, result)