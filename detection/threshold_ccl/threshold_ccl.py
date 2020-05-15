import cv2
import numpy as np





for i in range(1,28):
    original = cv2.imread("D:\\Users\\gavio\\CV30LperNick\detection\\threshold_ccl\\input\\{0:0=2d}.jpg".format(i), cv2.IMREAD_UNCHANGED)
    _, _, img = cv2.split(original)

    #img = cv2.medianBlur(img,5)
    #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,33,3)

    img = cv2.GaussianBlur(img,(5,5),1)
    _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, (5,5),iterations=3)
    retval, img, stats,_ = cv2.connectedComponentsWithStatsWithAlgorithm(img,8,cv2.CV_16U, cv2.CCL_GRANA)

    for j in range(retval):
        #check here
        top_left = (stats[j,cv2.CC_STAT_LEFT],stats[j,cv2.CC_STAT_TOP])
        bottom_right = (top_left[0] + stats[j,cv2.CC_STAT_WIDTH], top_left[1] +stats[j,cv2.CC_STAT_HEIGHT])
        img = cv2.rectangle(original, top_left,bottom_right,(255,0,0),2)

    cv2.imwrite("D:\\Users\\gavio\\CV30LperNick\detection\\threshold_ccl\\output\\{0:0=2d}.jpg".format(i), img) 
    

