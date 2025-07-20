from ultralytics import YOLO

# Load the custom model configuration
# We start from a pretrained model to leverage its learned weights,
# even though we have a custom head. The backbone weights are still valuable.
# Using yolov8n-pose.pt as a starting point.
model = YOLO('yolov8n-badminton.yaml').load('yolov8n-pose.pt')

# Train the model
if __name__ == '__main__':
    results = model.train(
        data='/Users/leeicheng/Documents/NCTU/MyYOLO/ultralytics/ultralytics/cfg/datasets/badminton_kpts.yaml',
        epochs=10,
        imgsz=640,
        project='runs/pose',
        name='badminton_kpts'
    )
