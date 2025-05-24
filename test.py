import torch, time

from ultralytics.models.yolo.model import GrayYOLO
from ultralytics import YOLO  # Ultralytics 內建接口

def forward_time(model, runs=20, size=640):
    # ── 選裝置：優先 MPS，其次 CPU ───────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        sync = getattr(torch.mps, "synchronize", lambda: None)
    else:
        device = torch.device("cpu")
        sync = lambda: None

    # dummy 一律 3-channel，因為灰階層自己會轉 1-ch
    dummy = torch.randn(1, 3, size, size, device=device)

    model = model.to(device).eval()

    with torch.no_grad():
        for _ in range(5):                               # 熱機
            model(dummy); sync()
        t0 = time.perf_counter()
        for _ in range(runs):                            # 正式計時
            model(dummy); sync()
        dt = (time.perf_counter() - t0) / runs

    return dt, str(device)


# --- 示範跑原版 / 灰階版 -----------------------------

plain = YOLO("yolov8n-pose.pt")
gray  = GrayYOLO("yolov8n-pose.pt")   # 你的灰階改裝

t_plain, dev = forward_time(plain)
t_gray,  _   = forward_time(gray)

print(f"跑在 {dev}")
print(f"plain : {t_plain*1000:.2f} ms")
print(f"gray  : {t_gray*1000:.2f} ms  ({(1-t_gray/t_plain)*100:.1f}% faster)")
