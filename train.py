from ultralytics import YOLO
import cv2

from ultralytics.models.yolo.model import GrayYOLO

model = GrayYOLO('yolov8n-pose.pt')
model.train(data="./datasets/dataset.yaml",
                   epochs=1,
                   batch=1,
                   imgsz=640)