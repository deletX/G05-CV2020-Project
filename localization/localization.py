def localization(painting_bboxes):
    votes = [0 for i in range(22)]
    for bbox in painting_bboxes:
        votes[bbox["painting"]["room"]] += 1
    return votes.index(max(votes))
