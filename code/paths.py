import os

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEOS_ROOT = os.path.join(PROJ_ROOT, "project_material", "videos")
MSF_DATASET = os.path.join(PROJ_ROOT, "msf_dataset")
IMAGE_DB = os.path.join(PROJ_ROOT, "retrieval", "paintings_db")
YOLO_WEIGHTS_PATH = os.path.join(PROJ_ROOT, "detection", "people", "yolo-coco", "yolov3.weights")
YOLO_CFG_PATH = os.path.join(PROJ_ROOT, "detection", "people", "yolo-coco", "yolov3.cfg")
