import cv2
import numpy as np
import glob
import json


def load_all_image_from_path(path):
    image_list = []
    image_names = []
    for filename in glob.glob(path):
        im = cv2.imread(filename)
        image_list.append(im)
        name = filename.split('\\')[1]
        image_names.append(name)
    return image_list, image_names


def load_json_file_from_path(path):
    with open(path) as json_file:
        content = json.load(json_file)
    return content


def convert_kps(keypoints):
    kps = []
    for el in keypoints:
        kp = cv2.KeyPoint(x=el[0], y=el[1], _size=el[2], _angle=el[3],
                          _response=el[4], _octave=el[5], _class_id=el[6])
        kps.append(kp)
    return kps


def convert_dscs(descriptors):
    return np.asarray(descriptors, dtype=np.uint8)


def convert_image(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_planes = cv2.split(rgb)
    planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        planes.append(norm_img)
    result = cv2.merge(planes)
    return result


def resize_images(q_image, t_image):
    qH, qW, _ = q_image.shape
    tH, tW, _ = t_image.shape
    if tH < qH:
        rH = qH
    else:
        rH = tH
    if tW < qW:
        rW = qW
    else:
        rW = tW
    query_image = cv2.resize(q_image, (rH, rW))
    train_image = cv2.resize(t_image, (rH, rW))
    return query_image, train_image


def retrieval(train_image, database):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #train_image = cv2.GaussianBlur(train_image, (7, 7), 0)
    #train_image = convert_image(train_image)
    results = []
    for query in database:
        query_image = cv2.imread('./paintings_db/' + query['image'])
        #query_image = convert_image(query_image)
        query_kps = query['keypoints']
        query_kps = convert_kps(query_kps)
        query_dscs = query['descriptors']
        query_dscs = convert_dscs(query_dscs)
        query_image, train_image = resize_images(query_image, train_image)
        train_gray = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)
        train_kps, train_dscs = orb.detectAndCompute(train_gray, None)
        if train_dscs is not None:
            matches = bf.match(query_dscs, train_dscs)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = []
            for match in matches:
                query_idx = match.queryIdx
                train_idx = match.trainIdx
                query_pt = query_kps[query_idx].pt
                train_pt = train_kps[train_idx].pt
                if abs(query_pt[0] - train_pt[0]) < 70 and abs(query_pt[1] - train_pt[1]) < 70:
                    good_matches.append(match)
            if good_matches:
                avg_distance = sum(match.distance for match in matches) / len(matches)
            else:
                avg_distance = 1000
            if len(matches) > 50 and len(good_matches) / len(matches) > 0.2:
                results.append((query_image, query_kps, train_image, train_kps, good_matches, avg_distance))
            good_matches = []
    if results:
        results = sorted(results, key=lambda x: x[5])[:10]
        return results


def main():
    train_images, images_names = load_all_image_from_path \
        ("C:/Users/marco/PycharmProjects/ProgettoCVxNik/msf_lillo/*.jpg")
    query_images = load_json_file_from_path("./paintings_descriptors.json")
    bboxs = load_json_file_from_path("C:/Users/marco/PycharmProjects/ProgettoCVxNik/rectification/rect_bboxs.json")
    bbox_images = []
    for index, image in enumerate(train_images):
        image_bboxs = bboxs[images_names[index]]
        for bbox in image_bboxs:
            bbox_images.append(image[bbox['y']:bbox['y'] + bbox['height'], bbox['x']:bbox['x'] + bbox['width']])
    for i, bbox_image in enumerate(bbox_images):
        result = retrieval(bbox_image, query_images)
        if result:
            for j, el in enumerate(result):
                j = j + 1
                img = cv2.drawMatches(el[0], el[1], el[2], el[3], el[4], None, flags=2)
                image = cv2.resize(img, (500, 500))
                cv2.imshow('Match: {} {}'.format(i, j), image)
                cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
