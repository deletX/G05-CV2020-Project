import cv2
import numpy as np
import glob
import json

from detection.threshold_ccl.threshold_ccl import run_frame
from rectification.rectification import rect


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
    return np.asarray(descriptors, dtype=np.float32)


def resize(q_image, t_image):
    qH, qW, _ = q_image.shape
    train_image = cv2.resize(t_image, (qW, qH))
    return train_image


def retrieval(train_image, database):
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    results = []
    for query in database:
        query_title = query['title']
        query_author = query['author']
        query_room = query['room']
        query_image = cv2.imread('./retrieval/paintings_db/' + query['image'])
        query_kps = query['keypoints']
        query_kps = convert_kps(query_kps)
        query_dscs = query['descriptors']
        query_dscs = convert_dscs(query_dscs)
        train_image = resize(query_image, train_image)
        train_gray = cv2.cvtColor(train_image, cv2.COLOR_RGB2GRAY)
        train_kps, train_dscs = sift.detectAndCompute(train_gray, mask=None)
        if train_dscs is not None and len(train_dscs) >= 2:
            matches = bf.match(query_dscs, train_dscs)
            good_matches = []
            for match in matches:
                query_idx = match.queryIdx
                train_idx = match.trainIdx
                query_pt = query_kps[query_idx].pt
                train_pt = train_kps[train_idx].pt
                if abs(query_pt[0] - train_pt[0]) < 20 and abs(query_pt[1] - train_pt[1]) < 20:
                    good_matches.append(match)
            if len(good_matches) > 0:
                results.append((query_title, query_author, query_room, query['image'], len(good_matches)))
    if len(results) > 0:
        results = sorted(results, key=lambda x: x[4], reverse=True)[:10]
        return results


def main():
    # images = load_json_file_from_path("../rectification/rect_bboxs.json")
    query_images = load_json_file_from_path("./paintings_descriptors.json")
    for i in range(1, 28):
        original = cv2.imread("../msf_lillo/{0:0=2d}.jpg".format(i),
                              cv2.IMREAD_UNCHANGED)
        bbox_list, img = run_frame(original)
        rected, bbox_rect = rect(original, bbox_list)
        for bbox in bbox_rect:
            train_image = bbox['rect']
            train_image = np.asarray(train_image, dtype=np.uint8)
            results = retrieval(train_image, query_images)
            if results:
                train_image = cv2.resize(train_image, (500, 500))
                cv2.imshow('Image', train_image)
                for i, res in enumerate(results):
                    print('{}) Title: {}, Author: {}, Room: {}, Image: {}'.format(i + 1, res[0],
                                                                                  res[1], res[2], res[3]))
                cv2.waitKey()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
