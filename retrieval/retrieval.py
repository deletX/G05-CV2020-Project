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
    return np.asarray(descriptors, dtype=np.float32)


def resize(q_image, t_image):
    qH, qW, _ = q_image.shape
    train_image = cv2.resize(t_image, (qW, qH))
    return train_image


def retrieval(train_image, database):
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    #index_params = dict(algorithm=0, trees=5)
    #search_params = dict()
    #flann = cv2.FlannBasedMatcher(index_params, search_params)
    results = []
    for query in database:
        query_image = cv2.imread('./paintings_db/' + query['image'])
        query_kps = query['keypoints']
        query_kps = convert_kps(query_kps)
        query_dscs = query['descriptors']
        query_dscs = convert_dscs(query_dscs)
        train_image = resize(query_image, train_image)
        train_gray = cv2.cvtColor(train_image, cv2.COLOR_RGB2GRAY)
        train_kps, train_dscs = sift.detectAndCompute(train_gray, mask=None)
        if train_dscs is not None and len(train_dscs) >= 2 and len(query_dscs) >= 2:
            matches = bf.match(query_dscs, train_dscs)
            #matches = flann.knnMatch(query_dscs, train_dscs, k=2)
            good_matches = []
            #for m, n in matches:
                #if m.distance < 0.8 * n.distance:
                    #good_matches.append(m)
            #good = []
            for match in matches:
                query_idx = match.queryIdx
                train_idx = match.trainIdx
                query_pt = query_kps[query_idx].pt
                train_pt = train_kps[train_idx].pt
                if abs(query_pt[0] - train_pt[0]) < 40 and abs(query_pt[1] - train_pt[1]) < 40:
                    good_matches.append(match)
            if len(good_matches) > 10:
                results.append((query_image, query_kps, train_image, train_kps, good_matches, len(good_matches)))
    results = sorted(results, key=lambda x: x[5], reverse=True)
    results = results[:10]
    if len(results) == 0:
        return None
    return results


def main():
    bboxs = load_json_file_from_path("../rectification/rect_bboxs.json")
    query_images = load_json_file_from_path("./paintings_descriptors.json")
    for key in bboxs:
        image = bboxs[key]
        for i, el1 in enumerate(image):
            train_image = el1['rect']
            train_image = np.asarray(train_image, dtype=np.uint8)
            result = retrieval(train_image, query_images)
            if result:
                for j, el2 in enumerate(result):
                    j = j + 1
                    img = cv2.drawMatches(el2[0], el2[1], el2[2], el2[3], el2[4], None, flags=2)
                    image = cv2.resize(img, (500, 500))
                    cv2.imshow('Match: {} {}'.format(i, j), image)
                    cv2.waitKey()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()