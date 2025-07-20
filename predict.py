from ultralytics import YOLO

# Load the trained custom model
# Replace 'path/to/your/trained/weights/best.pt' with the actual path to your best model weights.
# This path is typically found in the 'runs/pose/badminton_kpts/weights/' directory after training.
model = YOLO('runs/pose/badminton_kpts/weights/best.pt')

# Run prediction on an image
# Replace 'path/to/your/image.jpg' with the path to an image you want to predict on.
results = model('path/to/your/image.jpg')

# Parse and print the results
for result in results:
    if result.keypoints is not None:
        print(f"Detected {len(result.keypoints.xy)} keypoints.")
        keypoints_xy = result.keypoints.xy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for i in range(len(class_ids)):
            class_id = int(class_ids[i])
            class_name = model.names[class_id]
            x, y = keypoints_xy[i, 0, :]
            print(f"  - {class_name} at ({x:.2f}, {y:.2f})")
