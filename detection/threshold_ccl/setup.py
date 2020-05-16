import cv2

for i in range(1, 28):
    img = cv2.imread("C:/Users/marco/PycharmProjects/CV30LperNick/msf_lillo/{0:0=2d}.jpg".format(i),
                     cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    scale_percent = 100  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite("./input/{0:0=2d}.jpg".format(i), img)
