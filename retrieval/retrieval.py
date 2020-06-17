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


def retrieval(train_image, database):
    orb = cv2.ORB_create()
    train_gray = cv2.cvtColor(train_image, cv2.COLOR_RGB2GRAY)
    train_kps, train_dscs = orb.detectAndCompute(train_gray, mask=None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    smallest_detected_sum_of_distances = 100000000000.0
    results = []
    if train_dscs is not None:
        for query in database:
            query_image = cv2.imread('./paintings_db/' + query['image'])
            query_kps = query['keypoints']
            query_kps = convert_kps(query_kps)
            query_dscs = query['descriptors']
            query_dscs = convert_dscs(query_dscs)
            matches = bf.match(train_dscs, query_dscs)
            print(len(matches))
            distance_sum = sum(match.distance for match in matches)
            if distance_sum < smallest_detected_sum_of_distances:
                smallest_detected_sum_of_distances = distance_sum
                results.append((train_image, train_kps, query_image, query_kps, matches, distance_sum))
    results = sorted(results, key=lambda x: x[5])
    results = results[:10]
    if len(results) == 0:
        return None
    return results


def main():
    train_images, images_names = load_all_image_from_path\
        ("C:/Users/marco/PycharmProjects/ProgettoCVxNik/msf_lillo/*.jpg")
    train_images = train_images[:20]
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
                image = cv2.resize(img, (300, 300))
                cv2.imshow('Match: {} {}'.format(i, j), image)
                cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()