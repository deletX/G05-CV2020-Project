import cv2
import numpy as np
import json

# def run():
#     bboxes = {}
#
#     for i in range(1, 28):
#
#         original = cv2.imread("detection\\threshold_ccl\\input\\{0:0=2d}.jpg".format(i),
#                             cv2.IMREAD_UNCHANGED)
#
#         cv2.imwrite("./output/{0:0=2d}.jpg".format(i), img)
#         bboxes["{0:0=2d}.jpg".format(i)] = img_list
#
#     print(bboxes)
#     with open("bboxes_tccl.json","w") as out:
#         json_obj = json.dumps(bboxes)
#         out.write(json_obj)

def run_frame(hsv_frame):
    img_list = []
    _, _, img = cv2.split(hsv_frame)
        
    # img = cv2.medianBlur(img,5)
    # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,33,3)

    img = cv2.GaussianBlur(img, (7, 7), 1)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)

    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, (11, 11), iterations=3)
    retval, img, stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(img, 8, cv2.CV_16U, cv2.CCL_GRANA)

    for j in range(retval):
        top_left = (stats[j, cv2.CC_STAT_LEFT], stats[j, cv2.CC_STAT_TOP])
        bottom_right = (top_left[0] + stats[j, cv2.CC_STAT_WIDTH], top_left[1] + stats[j, cv2.CC_STAT_HEIGHT])
        img_list.append({
            "x":int(top_left[0]),
            "y":     int(top_left[1]),
            "width": int(bottom_right[0]-top_left[0]),
            "height":int(bottom_right[1]-top_left[1])
            })
        img = cv2.rectangle(hsv_frame, top_left, bottom_right, (255, 0, 0), 2)
    
    return img_list, img