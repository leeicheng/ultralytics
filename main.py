import time
import generate_dataset
from ultralytics import YOLO
import paths
import preview

model = YOLO('yolov8x-pose.pt')
dir_path = "./datasets"

start_time = time.time()

csv_paths,ball_paths = paths.find_all_csv_files(dir_path)
video_paths,pose_paths = paths.find_all_video_files(dir_path)
_, preview_video_paths = paths.create_preview_video_paths(dir_path)
_, crop_region_paths = paths.create_crop_region_paths(dir_path)
_, output_paths = paths.create_multi_output_paths(dir_path)
# ball's csv 轉換 為 json
generate_dataset.generate_ball_dataset(csv_paths,ball_paths)

first_crop_region = (240, 150, 1000, 720)  # (x1, y1, x2, y2)

for index in range(len(video_paths)):
    video_path = video_paths[index]
    pose_path = pose_paths[index]
    ball_path = ball_paths[index]
    output_path = output_paths[index]
    preview_video_path = preview_video_paths[index]
    crop_region_path = crop_region_paths[index]
    print(f"video:{video_path}")
    print(f"pose data:{pose_path}")
    print(f"pose data:{crop_region_path}")
    print(f"ball data:{ball_path}")
    print(f"output:{output_path}")

    generate_dataset.generate_pose_dataset(model, first_crop_region, video_path, pose_path)
    ## marge ball and pose json
    generate_dataset.generate_multi_dataset(ball_path, pose_path,output_path)
    ## preview result
    preview.generate_video(video_path,output_path,crop_region_path,preview_video_path,first_crop_region)

print(f"task done. {time.time() - start_time:.2f} seconds.")