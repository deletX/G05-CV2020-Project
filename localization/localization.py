def localization(painting_bboxes):
    votes = [0 for i in range(22)]
    for bbox in painting_bboxes:
        if "painting" in bbox:
            votes[int(bbox["painting"]["room"]) - 1] += 1
    return votes.index(max(votes)) + 1
