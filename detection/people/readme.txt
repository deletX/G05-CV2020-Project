ATTENZIONE!
prima di lanciare il codice, scaricare i pesi pre trainati qui: https://pjreddie.com/media/files/yolov3.weights
e successivamente aggiungere il file dentro alla cartella yolo-coco

per lanciare da linea di comando:

yolo.py -> #launch with: python yolo.py --image images/<image>.jpg --yolo yolo-coco
yolo_video.py -> python yolo_video.py --input videos/VIRB0401_Trim2.mp4 --output output/paintings_only_ppl2.avi --yolo yolo-coco

