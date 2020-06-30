import os

import cv2
import numpy as np
import json
from code.detection.painting.painting_detection import run_frame
from code.paths import PROJ_ROOT, MSF_DATASET
from code.rectification.rectification import rect


def load_json_file_from_path(path):
    """
    Opens a json file from the given path and return its content

    :param path: json file path
    :type path: str

    :return: Json file content
    """
    with open(path) as json_file:
        content = json.load(json_file)
    return content


def retrieval(detected_image, database):
    """
    Finds the closest 10 matches in the database to a given image

    :param detected_image: Image to look for into the painting databse
    :param database: Painting descriptors database.
    :return: A list of the 10 closest matches
    :rtype: list
    """
    sift = cv2.xfeatures2d.SIFT_create()

    # we use the brute force matcher to match the detected_image to the ones in the database
    bf = cv2.BFMatcher(crossCheck=True)
    results = []

    for db_image in database:
        # extract title, author, room, descriptors and shape
        db_image_title = db_image['title']
        db_image_author = db_image['author']
        db_image_room = db_image['room']
        db_image_dscs = np.asarray(db_image['descriptors'], dtype=np.float32)
        db_image_shape = db_image['shape']

        # compute SIFT descriptors for the given image
        _, detected_descriptors = sift.detectAndCompute(detected_image, mask=None)

        if detected_descriptors is not None:
            # if there are descriptors compute the brute force match onto the current database image,
            # compute the average distance of the matches and then add title, author, room,
            # image name and average distance to the results list
            matches = bf.match(db_image_dscs, detected_descriptors)
            avg_distance = sum(match.distance for match in matches) / len(matches)
            results.append((db_image_title, db_image_author, db_image_room, db_image['image'], avg_distance))
    if len(results) > 0:
        # if there are any result they get sorted o the average distance and the first 10 are returned
        results = sorted(results, key=lambda result: result[4])[:10]
        return results


def main():
    """
    Main function used for testing purposes
    :return:
    """
    query_images = load_json_file_from_path(os.path.join(PROJ_ROOT, "retrieval", "paintings_descriptors.json"))

    for i in range(1, 78):
        print("{0:0=2d}.jpg".format(i))
        original = cv2.imread(os.path.join(MSF_DATASET, "{0:0=2d}.jpg".format(i)), cv2.IMREAD_UNCHANGED)

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
