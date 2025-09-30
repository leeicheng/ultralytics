from ultralytics import YOLO


model = YOLO("yolov8-point.yaml").load("yolov8m.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco8.yaml", epochs=5, imgsz=640)
