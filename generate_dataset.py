import csv
import math
import json
import time

import cv2
import numpy as np

# 門檻設定
MAX_STATIC_SPEED = 2.0    # 平均每幀移動 ≤2px 就視為靜止
MAX_MATCH_DIST = 50.0     # 兩個框 center 距離 ≤50px 才算同一人
MIN_TRACK_LEN = 3

def get_centroid(bb):
    """從 {'X','Y','Width','Height'} 算中心點"""
    return np.array([bb['X'] + bb['Width'] / 2, bb['Y'] + bb['Height'] / 2])

def generate_multi_dataset(ball_path, pose_path,output_path):
    balls = json.load(open(ball_path))
    poses = json.load(open(pose_path))

    # if len(poses) != len(balls):
    #     # throw error
    #     return

    for index in range(len(poses)):
        pose = poses[index]
        frame_balls = []
        if index >= len(balls) - 1:
            ball_x = -1
            ball_y = -1
            frame_balls.append({"X": ball_x, "Y": ball_y})
            pose["Balls"] = frame_balls
            continue
        ball = balls[index]
        ball_x = ball["X"]
        ball_y = ball["Y"]
        frame_balls.append({"X": ball_x, "Y": ball_y})
        pose["Balls"] = frame_balls

    output = refactor_output(poses)

    # 2. track list，存每隻 track 的歷史中心點
    tracks = []  # 每個 track: { 'id', 'history': [centroids], 'instances': [(frame_idx, player_idx)] }
    next_id = 0

    for fi, frame in enumerate(output):
        centroids = [get_centroid(p['Bounding Box']) for p in frame['Players']]
        if fi == 0:
            # 首幀：全部新 track
            for pi, cen in enumerate(centroids):
                tracks.append({
                    'id': next_id,
                    'history': [cen],
                    'instances': [(fi, pi)]
                })
                next_id += 1
        else:
            assigned = set()
            # 1-to-1 greedy matching
            for tr in tracks:
                last = tr['history'][-1]
                dists = [np.linalg.norm(cen - last) for cen in centroids]
                pairs = sorted(enumerate(dists), key=lambda x: x[1])
                for pi, dist in pairs:
                    if pi in assigned or dist > MAX_MATCH_DIST:
                        continue
                    # assign detection to track
                    tr['history'].append(centroids[pi])
                    tr['instances'].append((fi, pi))
                    assigned.add(pi)
                    break
            # 剩餘沒配對到的，當新 track
            for pi in range(len(centroids)):
                if pi not in assigned:
                    tracks.append({
                        'id': next_id,
                        'history': [centroids[pi]],
                        'instances': [(fi, pi)]
                    })
                    next_id += 1

    # 先把短少的 track 直接踢掉
    tracks = [tr for tr in tracks if len(tr['history']) >= MIN_TRACK_LEN]

    # 3. 判別哪個 track 在移動
    moving_ids = set()
    for tr in tracks:
        hist = tr['history']
        if len(hist) < 2:
            continue

        dists = [np.linalg.norm(hist[i] - hist[i - 1]) for i in range(1, len(hist))]
        avg_speed = sum(dists) / len(dists)
        if avg_speed > MAX_STATIC_SPEED:
            moving_ids.add(tr['id'])

    # 4. 產生新的 frames
    cleaned = []
    for fi, frame in enumerate(output):
        new_players = []
        for tr in tracks:
            for fidx, pidx in tr['instances']:
                if fidx == fi and tr['id'] in moving_ids:
                    new_players.append(frame['Players'][pidx])
        cleaned.append({
            'Frame': frame['Frame'],
            'Balls': frame.get('Balls', []),
            'Players': new_players
        })

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(cleaned, file, indent=2)

def generate_ball_dataset(csv_paths: str,output_paths: str):
    """
    Convert all CSV files in the provided directory to JSON.

    Args:
        root_dir_path (str): The directory containing CSV files.
    """
    for index,csv_path in enumerate(csv_paths):
        output = convert_csv_to_json(csv_path)
        output_path = output_paths[index]
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(output, file, indent=2)
    print("完成！已將所有 CSV 轉為 JSON。")

def generate_pose_dataset(model,crop_region,video_path,output_path):
    start_time = time.time()
    # ----- 初始化 -----
    cap = cv2.VideoCapture(video_path)

    all_results = []
    frame_id = 0

    # ----- 處理每一幀 -----
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 裁切畫面
        cropped_frame = frame[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]
        annotated_frame = cropped_frame.copy()

        results = model(annotated_frame, conf=0.5)[0]

        boxes = results.boxes.xywh.cpu().numpy()  # (x_center, y_center, w, h)
        keypoints = results.keypoints.xy.cpu().numpy()  # (n, 17, 2)
        players = []

        for i, kpts in enumerate(keypoints):
            # 調整 keypoints 座標為整張圖的位置
            kpts_list = []
            for x, y in kpts:
                x_full = float(x) + crop_region[0] if x != 0 else 0
                y_full = float(y) + crop_region[1] if y != 0 else 0

                # x_full = float(x) if x != 0 else 0
                # y_full = float(y) if y != 0 else 0
                kpts_list.append({"X": int(x_full), "Y": int(y_full)})

            # 處理 bounding box
            x_center, y_center, w, h = boxes[i]
            x_full = int(x_center + crop_region[0])
            y_full = int(y_center + crop_region[1])
            # x_full = int(x_center)
            # y_full = int(y_center)

            bbox = {
                "X": x_full,
                "Y": y_full,
                "Width": int(w),
                "Height": int(h)
            }

            # 畫框（畫在 cropped 上）
            # x1 = int(x_center - w / 2)
            # y1 = int(y_center - h / 2)
            # x2 = int(x_center + w / 2)
            # y2 = int(y_center + h / 2)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # 加入單一 player 結構
            players.append({
                "Bounding Box": bbox,
                "Keypoints": kpts_list
            })

        # 組裝 JSON 結構
        all_results.append({
            "Frame": frame_id,
            "Players": players
        })

        frame_id += 1

    # ----- 收尾 -----
    cap.release()

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"完成！已產出影片與格式化 JSON。{time.time() - start_time:.2f} seconds.")

def generate_pose_dataset_batch(model, crop_region, video_path, output_path, batch_size=4):
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    all_results = []
    frame_id = 0
    batch_frames = []
    batch_raw_frames = []
    batch_frame_ids = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped_frame = frame[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]
        batch_frames.append(cropped_frame)
        batch_raw_frames.append(frame)
        batch_frame_ids.append(frame_id)
        frame_id += 1

        if len(batch_frames) == batch_size:
            all_results.extend(process_batch(model, batch_frames, batch_raw_frames, batch_frame_ids, crop_region))
            batch_frames = []
            batch_raw_frames = []
            batch_frame_ids = []

    # 收尾：剩下不足 batch 的也要處理
    if batch_frames:
        all_results.extend(process_batch(model, batch_frames, batch_raw_frames, batch_frame_ids, crop_region))

    cap.release()
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"完成！已產出影片與格式化 JSON。{time.time() - start_time:.2f} seconds.")


def process_batch(model, cropped_batch, raw_batch, frame_ids, crop_region):
    results_batch = model(cropped_batch, conf=0.5)
    batch_results = []

    for b_idx, results in enumerate(results_batch):
        frame_results = []
        boxes = results.boxes.xywh.cpu().numpy()
        keypoints = results.keypoints.xy.cpu().numpy()

        if len(boxes) == 0 or len(keypoints) == 0:
            batch_results.append({
                "Frame": frame_ids[b_idx],
                "Players": []  # 空的
            })
            continue

        for i, kpts in enumerate(keypoints):
            kpts_list = []
            for x, y in kpts:
                x_full = float(x) + crop_region[0] if x != 0 else 0
                y_full = float(y) + crop_region[1] if y != 0 else 0
                kpts_list.append({"X": int(x_full), "Y": int(y_full)})

            x_center, y_center, w, h = boxes[i]
            x_full = int(x_center + crop_region[0])
            y_full = int(y_center + crop_region[1])

            bbox = {
                "X": x_full,
                "Y": y_full,
                "Width": int(w),
                "Height": int(h)
            }

            frame_results.append({
                "Bounding Box": bbox,
                "Keypoints": kpts_list
            })

        batch_results.append({
            "Frame": frame_ids[b_idx],
            "Players": frame_results
        })

    return batch_results

def generate_pose_dataset_batch_with_dynamic_crop(model, crop_region, crop_out_path, video_path, output_path, batch_size=4):
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    all_results = []
    crop_regions_log = []  # 新增：記錄每幀 crop_region

    frame_id = 0
    batch_frames = []
    batch_raw_frames = []
    batch_frame_ids = []

    prev_crop_region = crop_region  # 初始化 crop 區域

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 使用動態 crop_region
        x1, y1, x2, y2 = prev_crop_region
        cropped_frame = frame[y1:y2, x1:x2]

        batch_frames.append(cropped_frame)
        batch_raw_frames.append(frame)
        batch_frame_ids.append(frame_id)
        frame_id += 1

        if len(batch_frames) == batch_size:
            all_results_batch, prev_crop_region = process_batch_with_tracking(
                model, batch_frames, batch_raw_frames, batch_frame_ids, prev_crop_region
            )
            all_results.extend(all_results_batch)

            # 記錄每一幀 crop_region
            for _ in batch_frame_ids:
                crop_regions_log.append({"crop_region": list(prev_crop_region)})

            batch_frames = []
            batch_raw_frames = []
            batch_frame_ids = []

    if batch_frames:
        all_results_batch, _ = process_batch_with_tracking(
            model, batch_frames, batch_raw_frames, batch_frame_ids, prev_crop_region
        )
        all_results.extend(all_results_batch)

        for _ in batch_frame_ids:
            crop_regions_log.append({"crop_region": list(prev_crop_region)})

    cap.release()

    #  存 JSON 結果
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    #  存 crop_region 紀錄
    with open(crop_out_path, 'w') as f:
        json.dump(crop_regions_log, f, indent=2)

    print(f"完成！已產出影片與格式化 JSON。{time.time() - start_time:.2f} seconds.")
    print(f"Crop region log saved to: {crop_out_path}")

def process_batch_with_tracking(model, cropped_batch, raw_batch, frame_ids, crop_region):
    results_batch = model(cropped_batch, conf=0.5)
    batch_results = []
    new_crop = [9999, 9999, 0, 0]  # 找新 crop 區域

    for b_idx, results in enumerate(results_batch):
        frame_results = []
        boxes = results.boxes.xywh.cpu().numpy()
        keypoints = results.keypoints.xy.cpu().numpy()

        is_retry = False
        use_original_result = True  # 預設保留第一次推理的結果

        # 判斷是否需要重推
        if len(boxes) == 1:
            print(f"Frame {frame_ids[b_idx]} has only 1 person, retrying with previous crop_region...")
            is_retry = True

            original_frame = raw_batch[b_idx]
            x1, y1, x2, y2 = crop_region
            re_cropped = original_frame[y1:y2, x1:x2]
            re_result = model([re_cropped], conf=0.5)[0]

            re_boxes = re_result.boxes.xywh.cpu().numpy()
            re_keypoints = re_result.keypoints.xy.cpu().numpy()

            # 若重推後人數大於 1，就用重推結果
            if len(re_boxes) > 1 and len(re_boxes) == len(re_keypoints):
                print(f"Retry successful for Frame {frame_ids[b_idx]} → using re-predict result")
                boxes = re_boxes
                keypoints = re_keypoints
                use_original_result = False
            else:
                print(f"Retry failed for Frame {frame_ids[b_idx]} → keep original result")

        # 若完全沒人 or 預測異常（只有 retry 也不通過）
        if len(boxes) == 0 or len(keypoints) == 0 or len(boxes) != len(keypoints):
            batch_results.append({
                "Frame": frame_ids[b_idx],
                "Players": []
            })
            continue

        for i, kpts in enumerate(keypoints):
            kpts_list = []
            for x, y in kpts:
                x_full = float(x) + crop_region[0] if x != 0 else 0
                y_full = float(y) + crop_region[1] if y != 0 else 0
                kpts_list.append({"X": int(x_full), "Y": int(y_full)})

            x_center, y_center, w, h = boxes[i]
            x_full = x_center + crop_region[0]
            y_full = y_center + crop_region[1]

            x1 = int(x_full - w / 2)
            y1 = int(y_full - h / 2)
            x2 = int(x_full + w / 2)
            y2 = int(y_full + h / 2)

            if not is_retry or not use_original_result:
                # 只有在「使用有效預測結果」時才更新 crop
                new_crop[0] = min(new_crop[0], x1)
                new_crop[1] = min(new_crop[1], y1)
                new_crop[2] = max(new_crop[2], x2)
                new_crop[3] = max(new_crop[3], y2)

            bbox = {
                "X": int(x_full),
                "Y": int(y_full),
                "Width": int(w),
                "Height": int(h)
            }

            frame_results.append({
                "Bounding Box": bbox,
                "Keypoints": kpts_list
            })

        batch_results.append({
            "Frame": frame_ids[b_idx],
            "Players": frame_results
        })

    # 更新 crop_region（只在有效推理情況下才會變動）
    if new_crop[0] < new_crop[2] and new_crop[1] < new_crop[3]:
        pad = 20
        new_crop = [
            max(new_crop[0] - pad, 0),
            max(new_crop[1] - pad, 0),
            new_crop[2] + pad,
            new_crop[3] + pad,
        ]
    else:
        new_crop = crop_region

    return batch_results, new_crop

def convert_csv_to_json(file_path: str) -> dict:
    """
    Convert a single CSV file to a JSON object.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        dict: The JSON representation of the CSV data.
    """
    header = []
    output = []
    with open(file_path, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row_index, row in enumerate(reader):
            if row_index == 0:  # 第一行作為 header
                header = list(row)
            else:
                dic = {}
                for index, data in enumerate(row):
                    # 將字串型數據轉換為 float 後向上取整
                    dic[header[index]] = math.ceil(float(data))
                output.append(dic)
    return output

def refactor_output(origin):
    refactor_rows = []
    for data in origin:
        refactor = {
            "Frame" : data["Frame"],
            "Balls" : data["Balls"],
            "Players" : data["Players"]
        }
        refactor_rows.append(refactor)
    return refactor_rows