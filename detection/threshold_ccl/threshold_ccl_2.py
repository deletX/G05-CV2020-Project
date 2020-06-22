import cv2
import json
import numpy as np
import time


def millis():
    return int(round(time.time() * 1000))


def run():
    bboxes = {}

    for i in range(1, 28):
        original = cv2.imread("../../msf_lillo/{0:0=2d}.jpg".format(i),
                              cv2.IMREAD_UNCHANGED)
        img_list, img = run_frame(original)
        print("../../msf_lillo/{0:0=2d}.jpg".format(i))
        cv2.imwrite("./output2/{0:0=2d}.jpg".format(i), img)
        bboxes["{0:0=2d}.jpg".format(i)] = img_list

    print(bboxes)
    with open("./output2/bboxes_tccl.json", "w") as out:
        json_obj = json.dumps(bboxes)
        out.write(json_obj)


def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1, a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]


def run_frame(frame):
    frame = cv2.resize(frame, (1280, 720))

    img_list = []

    spatial_radius = 10
    color_radius = 13
    maximum_pyramid_level = 1

    blur = cv2.GaussianBlur(frame, (7, 7), 1)
    mean_shift = cv2.pyrMeanShiftFiltering(blur, spatial_radius, color_radius, maximum_pyramid_level)
    gray = cv2.cvtColor(mean_shift, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    cl1 = clahe.apply(gray)
    _, thresh = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, (15, 15), iterations=3)
    inv = cv2.bitwise_not(close)

    retval, _, stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(inv, 8, cv2.CV_16U, cv2.CCL_GRANA)

    for j in range(retval):
        top_left = (stats[j, cv2.CC_STAT_LEFT], stats[j, cv2.CC_STAT_TOP])
        bottom_right = (top_left[0] + stats[j, cv2.CC_STAT_WIDTH], top_left[1] + stats[j, cv2.CC_STAT_HEIGHT])
        width = int(bottom_right[0] - top_left[0])
        height = int(bottom_right[1] - top_left[1])

        # real_area = stats[j, cv2.CC_STAT_AREA]
        area = width * height
        if 32000 < area < 1280 * 720:
            crop = inv[top_left[1]:top_left[1] + height, top_left[0]:top_left[0] + width]
            close2 = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, (7, 7), iterations=2)
            median = cv2.medianBlur(close2, 5)
            img = np.uint8(median)
            _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cv2.fillPoly(median, contour, color=(255, 255, 255))

            # cv2.imshow("test", median)
            # cv2.waitKey()
            img_list.append({
                "x": int(top_left[0]),
                "y": int(top_left[1]),
                "width": width,
                "height": height,

            })
            img = cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
        else:
            img = frame
    # cv2.imshow("test", img)
    # cv2.waitKey()
    return img_list, img


if __name__ == "__main__":
    start = millis()
    run()
    end = millis()
    print("Took: {}ms".format(end - start))
