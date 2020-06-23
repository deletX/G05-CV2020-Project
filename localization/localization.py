def localization(painting_bboxes):
    """
    Perform the people localization using a basic voting system

    :param painting_bboxes: Dictionary containing the painting bounding boxes for which the retrieved painting has been added
    :type painting_bboxes: dict
    :return: The detected room
    :rtype: int
    """
    # initialize the possible votes (22 rooms)
    votes = [0 for i in range(22)]
    for bbox in painting_bboxes:

        # if in the bounding box has been found a painting by the retrieval step
        if "painting" in bbox:
            # increment the votes for the room in which such painting is located
            votes[int(bbox["painting"]["room"]) - 1] += 1

    # return the index + 1  (room) that has received the maximum number of votes.
    # (the first if there are more with the same number of votes)
    return votes.index(max(votes)) + 1
