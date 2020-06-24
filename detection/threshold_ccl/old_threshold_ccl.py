import json
import cv2
import numpy as np


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
    cv2.imshow("value", img)

    img = cv2.GaussianBlur(img, (7, 7), 1)
    cv2.imshow("blur", img)

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
    cv2.imshow("threshold", img)

    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, (11, 11), iterations=3)
    cv2.imshow("morph", img)

    retval, img, stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(img, 8, cv2.CV_16U, cv2.CCL_GRANA)

    label_hue = np.uint8(179 * img / np.max(img))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0
    cv2.imshow('labeled', labeled_img)

    for j in range(retval):
        top_left = (stats[j, cv2.CC_STAT_LEFT], stats[j, cv2.CC_STAT_TOP])
        bottom_right = (top_left[0] + stats[j, cv2.CC_STAT_WIDTH], top_left[1] + stats[j, cv2.CC_STAT_HEIGHT])
        width = int(bottom_right[0] - top_left[0])
        height = int(bottom_right[1] - top_left[1])
        area = width * height
        if area > 32000:
            img_list.append({
                "x": int(top_left[0]),
                "y": int(top_left[1]),
                "width": width,
                "height": height
            })
            img = cv2.rectangle(hsv_frame, top_left, bottom_right, (255, 0, 0), 2)
        else:
            img = hsv_frame
    return img_list, img


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
        _, img = run_frame(frame)
        cv2.imshow("Result", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main_check_img(frame_path):
    frame = cv2.imread(frame_path, cv2.COLOR_BGR2HSV)
    cv2.imshow("Result", frame)
    _, img = run_frame(frame)
    cv2.imshow("Result", img)
    cv2.waitKey()


if __name__ == "__main__":
    main_check_img("../../msf_lillo/01.jpg")
