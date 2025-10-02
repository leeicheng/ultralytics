from ultralytics import YOLO


model = YOLO("yolov8-point.yaml").load("yolov8m.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/Users/leeicheng/Documents/NCTU/MyYOLO/ultralytics/datasets/dataset.yaml", epochs=5, imgsz=640)
