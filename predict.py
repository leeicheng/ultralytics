from ultralytics.models.yolo.model import GrayYOLO


model = GrayYOLO("./runs/pose/train10/weights/best.pt")          # or "runs/train/exp/weights/best.pt"

# ❷ 推論 (predict) ───────────────────────────────
results = model.predict(
    source="./datasets/img.png",          # 檔案/資料夾/影片/RTSP 都行
    imgsz=640,                   # 尺寸→內部仍吃 RGB，wrapper 會轉灰階
    conf=0.25,                   # 置信度
    save=True,                   # 把繪製結果存檔到 runs/predict/
)

# ❸ 驗證 (val) ────────────────────────────────
val_metrics = model.val(
    data="./datasets/dataset.yaml",  # 你的 dataset config
    imgsz=640,
    split="val",                     # 如果 YAML 裡有自訂 split
    half=True,                       # MPS 不支援 half；CUDA 才能開
)

print(val_metrics)   # dict 裡有 mAP, OKS_AP, PCK 等