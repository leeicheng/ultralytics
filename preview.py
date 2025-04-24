import os
import json
import math
import cv2

def generate_video(video_path, multi_path, crop_regin_path, output_path, crop_region):
    # 建立 VideoCapture 與讀取標籤/裁切區域
    source_video = cv2.VideoCapture(video_path)
    labels = json.load(open(multi_path, 'r', encoding='utf-8'))
    # crop_regin_json = json.load(open(crop_regin_path, 'r', encoding='utf-8'))  # 每幀 crop_region

    # 取得影片參數
    fps = int(source_video.get(cv2.CAP_PROP_FPS))
    video_w = int(source_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(source_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (video_w, video_h))

    # 準備儲存每幀影像的資料夾
    frames_dir = os.path.join(os.path.dirname(output_path), 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    frame_id = 0
    while True:
        ret, frame = source_video.read()
        if not ret or frame_id >= len(labels):
            break

        annotated_frame = frame.copy()
        label = labels[frame_id]
        balls = label.get('Balls', [])
        players = label.get('Players', [])

        # 根據每一幀的 crop_region 畫框
        # if frame_id < len(crop_regin_json):
        #     crop = crop_regin_json[frame_id].get('crop_region', crop_region)
        # else:
        crop = crop_region
        cv2.rectangle(
            annotated_frame,
            (int(crop[0]), int(crop[1])),
            (int(crop[2]), int(crop[3])),
            (0, 100, 255), 2
        )

        # 畫球
        if balls:
            ball = balls[0]
            cv2.circle(
                annotated_frame,
                (int(ball['X']), int(ball['Y'])), 3,
                (0, 0, 255), -1
            )

        # 畫玩家與關鍵點
        for player in players:
            box = player.get('Bounding Box', {})
            keypoints = player.get('Keypoints', [])
            x_c, y_c = box.get('X', 0), box.get('Y', 0)
            w, h = box.get('Width', 0), box.get('Height', 0)
            x1, y1 = int(x_c - w/2), int(y_c - h/2)
            x2, y2 = int(x_c + w/2), int(y_c + h/2)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            for kp in keypoints:
                cv2.circle(
                    annotated_frame,
                    (int(kp['X']), int(kp['Y'])), 3,
                    (0, 255, 255), -1
                )

        # 寫入影片
        out.write(annotated_frame)

        # 輸出單張影像到 frames 資料夾
        # frame_path = os.path.join(frames_dir, f'{frame_id}.jpg')
        # cv2.imwrite(frame_path, annotated_frame)

        frame_id += 1

    # 釋放資源
    source_video.release()
    out.release()
    print(f"完成！已產出影片，並將每幀儲存至 {frames_dir}")