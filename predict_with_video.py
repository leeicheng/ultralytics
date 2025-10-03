import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import os


def draw_circle_with_alpha(img, center, radius, color, confidence):
    """畫帶透明度的實心圓點，透明度依信心分數線性縮放。"""
    overlay = img.copy()
    cv2.circle(overlay, center, radius, color, -1)
    alpha = float(0.3 + 0.7 * max(0.0, min(1.0, confidence)))
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def process_video(model_path, video_path, output_path=None, save_frames=False, conf_thres=0.0):
    """
    使用 PointDetect 模型處理影片：讀取影片 -> 預測每一幀 -> 畫點與文字 -> 重新組合成影片

    Args:
        model_path (str): 訓練好的 .pt 模型檔案路徑
        video_path (str): 輸入影片路徑
        output_path (str): 輸出影片路徑 (可選)
        save_frames (bool): 是否儲存處理後的幀圖片
        conf_thres (float): 顯示的最小信心分數
    """

    # 載入模型
    try:
        model = YOLO(model_path)
        print(f"模型 '{model_path}' 載入成功。")
    except Exception as e:
        print(f"錯誤：無法載入模型。 {e}")
        return

    # 檢查影片檔案是否存在
    if not os.path.exists(video_path):
        print(f"錯誤：影片檔案不存在 -> {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片檔案 -> {video_path}")
        return

    # 影片資訊
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("影片資訊：")
    print(f"  - 解析度: {width}x{height}")
    print(f"  - FPS: {fps}")
    print(f"  - 總幀數: {total_frames}")
    if fps:
        print(f"  - 影片長度: {total_frames / max(fps,1):.2f} 秒")

    # 輸出路徑
    if output_path is None:
        video_stem = Path(video_path).stem
        output_path = f"{video_stem}_predicted.mp4"

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # 類別資訊（BGR 顏色, 半徑, 前綴）
    class_info = {
        0: {"color": (0, 0, 255), "radius": 6, "prefix": "T"},   # Red
        1: {"color": (255, 0, 0), "radius": 6, "prefix": "C"},  # Blue
        2: {"color": (0, 255, 0), "radius": 6, "prefix": "L"},  # Green
    }
    default_info = {"color": (255, 255, 255), "radius": 6, "prefix": "UNK"}

    # 幀圖片儲存
    frames_dir = None
    if save_frames:
        video_stem = Path(video_path).stem
        frames_dir = out_dir / f"{video_stem}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

    print("\n開始處理影片...")
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"處理第 {frame_count}/{total_frames} 幀", end="\r")

            # 使用 YOLO 模型進行預測（PointDetect 會回傳 Results，其中 keypoints、point_cls、point_conf 可用）
            results = model(frame, verbose=False)

            output_frame = frame.copy()

            for result in results:
                # PointDetect predictor 會回傳：
                # - result.keypoints: Keypoints 物件，包含 shape (N, 1, 3) 的 tensor
                # - result.point_cls: tensor (N,) -> 類別 ID
                # - result.point_conf: tensor (N,) -> 信心分數
                kpts = getattr(result, "keypoints", None)
                pcls = getattr(result, "point_cls", None)
                pconf = getattr(result, "point_conf", None)

                # 檢查是否有檢測結果
                if kpts is None or pcls is None or pconf is None:
                    continue

                # 檢查 keypoints 是否為空
                if not hasattr(kpts, 'data') or kpts.data.shape[0] == 0:
                    continue

                # 取得 xy 坐標 (N, 1, 2) 並轉換成 numpy
                xy_coords = kpts.xy.cpu().numpy()  # (N, 1, 2)

                # 處理 class_ids 和 confidences（可能是 tensor 或 list）
                if hasattr(pcls, 'cpu'):
                    class_ids = pcls.cpu().numpy().astype(int)
                else:
                    class_ids = np.array(pcls, dtype=int)

                if hasattr(pconf, 'cpu'):
                    confidences = pconf.cpu().numpy()
                else:
                    confidences = np.array(pconf, dtype=float)

                # 迭代每個檢測點
                for j in range(xy_coords.shape[0]):
                    x, y = xy_coords[j, 0, :]  # 取得第 j 個點的 (x, y)
                    cls_id = int(class_ids[j])
                    conf = float(confidences[j])

                    if conf < conf_thres:
                        continue

                    cx, cy = int(round(x)), int(round(y))
                    info = class_info.get(cls_id, default_info)
                    color = info["color"]
                    radius = int(info["radius"])
                    prefix = info["prefix"]

                    draw_circle_with_alpha(output_frame, (cx, cy), radius, color, conf)

                    # 文字標籤
                    label = f"{prefix}_{conf:.2f}"
                    font_face = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    font_thickness = 1
                    text_position = (cx + radius + 2, cy - radius - 2)
                    cv2.putText(output_frame, label, text_position, font_face, font_scale, color, font_thickness, cv2.LINE_AA)

            out.write(output_frame)

            if save_frames and frames_dir is not None:
                frame_filename = frames_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_filename), output_frame)

    except KeyboardInterrupt:
        print(f"\n處理被中斷，已處理 {frame_count} 幀")
    except Exception as e:
        print(f"\n處理過程中發生錯誤: {e}")
    finally:
        cap.release()
        out.release()
        print("\n影片處理完成！")
        print(f"輸出影片已儲存至: {output_path}")
        if save_frames and frames_dir is not None:
            print(f"處理後的幀圖片已儲存至: {frames_dir}")


def batch_process_videos(model_path, video_dir, output_dir=None, save_frames=False, conf_thres=0.0):
    """批次處理資料夾中的所有影片。"""
    video_dir = Path(video_dir)
    if output_dir is None:
        output_dir = video_dir / "processed_videos"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))

    if not video_files:
        print(f"在資料夾 '{video_dir}' 中沒有找到影片檔案")
        return

    print(f"找到 {len(video_files)} 個影片檔案")
    for i, video_file in enumerate(video_files, 1):
        print(f"\n--- 處理影片 {i}/{len(video_files)}: {video_file.name} ---")
        output_path = output_dir / f"{video_file.stem}_predicted.mp4"
        process_video(model_path, str(video_file), str(output_path), save_frames, conf_thres)

 # python predict_with_video.py --input "F:\NYCU\ultralytics\datasets\videos_grayscale" --output "F:\NYCU\ultralytics\datasets\output\pd_t19_g" --batch --save-frames
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 影片處理工具 - PointDetect 交點預測")
    parser.add_argument("--model", type=str, help="訓練好的模型權重(.pt)",default="F:/NYCU/ultralytics/runs/point/train19/weights/best.pt",)
    parser.add_argument("--input", type=str, required=True, help="輸入影片檔案或資料夾路徑")
    parser.add_argument("--output", type=str, default=None, help="輸出影片檔案或輸出資料夾路徑")
    parser.add_argument("--save-frames", action="store_true", help="是否儲存處理後的幀圖片")
    parser.add_argument("--batch", action="store_true", help="批次處理資料夾中的所有影片")
    parser.add_argument("--conf", type=float, default=0.9, help="顯示的最小信心分數")

    args = parser.parse_args()
    if args.batch:
        batch_process_videos(args.model, args.input, args.output, args.save_frames, args.conf)
    else:
        process_video(args.model, args.input, args.output, args.save_frames, args.conf)

    print("\n圖例:")
    print("  - 紅色實心圓: T-junction (類別 0)")
    print("  - 藍色實心圓: Cross-junction (類別 1)")
    print("  - 綠色實心圓: L-corner (類別 2)")
    print("  - 透明度代表信賴度：越透明信賴度越低")
