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
        kp = cv2.KeyPoint(x=el['point'][0], y=el['point'][1], _size=el['size'], _angle=el['angle'],
                          _response=el['response'], _octave=el['octave'], _class_id=el['class_id'])
        kps.append(kp)
    return kps


def convert_dscs(descriptors):
    return np.asarray(descriptors, dtype=np.uint8)


def display_image(image, title="image"):
    img = cv2.resize(image, (600, 600))
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def combine_image(image1, image2):
    h1, w1, c1 = image1.shape[:3]
    h2, w2, c2 = image2.shape[:3]
    vis = np.zeros((max(h1, h2), w1 + w2, c1), np.uint8)
    vis[:h1, :w1, :c1] = image1
    vis[:h2, w1:w1 + w2, :c2] = image2
    return vis


def compare_hists(train_image, query_image):
    q_image = cv2.imread('./paintings_db/' + query_image['image'])
    q_blue = cv2.calcHist([q_image], [0], None, [256], [0, 256])
    q_green = cv2.calcHist([q_image], [1], None, [256], [0, 256])
    q_red = cv2.calcHist([q_image], [2], None, [256], [0, 256])
    t_blue = cv2.calcHist([train_image], [0], None, [256], [0, 256])
    t_green = cv2.calcHist([train_image], [1], None, [256], [0, 256])
    t_red = cv2.calcHist([train_image], [2], None, [256], [0, 256])
    blue = cv2.compareHist(q_blue, t_blue, cv2.HISTCMP_INTERSECT)
    green = cv2.compareHist(q_green, t_green, cv2.HISTCMP_INTERSECT)
    red = cv2.compareHist(q_red, t_red, cv2.HISTCMP_INTERSECT)
    return (blue + green + red) / 3


def orb_with_flann(train_image, query_image):
    orb = cv2.ORB_create(nfeatures=2000)
    flann_index_lsh = 6
    index_params = dict(algorithm=flann_index_lsh,
                        table_number=12,
                        key_size=20,
                        multi_probe_level=2)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    train_kp, train_dscs = orb.detectAndCompute(train_image, None)
    image = query_image['image']
    title = query_image['title']
    author = query_image['author']
    room = query_image['room']
    dscs = query_image['descriptors']
    query_dscs = convert_dscs(dscs)
    if (query_dscs is not None and len(query_dscs) > 2 and train_dscs is not None and len(train_dscs) > 2):
        flann_matches = flann.knnMatch(query_dscs, train_dscs, k=2)
        matches_mask = [[0, 0] for i in range(len(flann_matches))]
        good = []
        for index in range(len(flann_matches)):
            if len(flann_matches[index]) == 2:
                m, n = flann_matches[index]
                if m.distance < 0.8 * n.distance:
                    matches_mask[index] = [1, 0]
                    good.append(flann_matches[index])
        return image, len(good), {"title": title, "author": author, "room": room}


def find_best_match_index(match):
    index_best = 0
    best_length = 0
    res = []
    for m in match:
        if m is not None:
            res.append(m)
    if not res:
        index_best = -1
        return index_best
    for index, (image, good_match_length, det) in enumerate(res):
        if good_match_length >= best_length:
            best_length = good_match_length
            index_best = index
    return index_best


def retrieval(frame, query_images, bbox_list):
    bbox_images = []
    for bbox in bbox_list:
        bbox_images.append(frame[bbox['y']:bbox['y'] + bbox['height'], bbox['x']:bbox['x'] + bbox['width']])
    for bbox_image in bbox_images:
        bbox_image_gray = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2GRAY)
        matches = []
        hist = None
        for query_image in query_images:
            matches.append(orb_with_flann(bbox_image_gray, query_image))
        best_match_index = find_best_match_index(matches)
        if best_match_index >= 0:
            bbox['painting'] = matches[best_match_index][2]
    return bbox_list


def main():
    train_images, images_names = load_all_image_from_path \
        ("../rectification/output/*.jpg")
    train_images = train_images
    query_images = load_json_file_from_path("./paintings_descriptors.json")
    bboxs = load_json_file_from_path("../rectification/rect_bboxs.json")
    bbox_images = []
    for index, image in enumerate(train_images):
        image_bboxs = bboxs[images_names[index]]
        for bbox in image_bboxs:
            bbox_images.append(image[bbox['y']:bbox['y'] + bbox['height'], bbox['x']:bbox['x'] + bbox['width']])
    for bbox_image in bbox_images:
        bbox = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2GRAY)
        matches = []
        hist = None
        for query_image in query_images:
            matches.append(orb_with_flann(bbox, query_image))
            # hist = compare_hists(bbox_image, query_image)
        best_match_index = find_best_match_index(matches)
        if best_match_index >= 0:
            retr_image = cv2.imread("./paintings_db/" + matches[best_match_index][0])
            result_image = combine_image(bbox_image, retr_image)
            display_image(result_image)


if __name__ == "__main__":
    main()
