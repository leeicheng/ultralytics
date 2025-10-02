from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8-point.yaml").load("yolov8m.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="F:/NYCU/ultralytics/datasets/dataset_config.yaml", epochs=100, imgsz=640)