"""
抓速度、密度、大車、小車
"""

from speed.speed_estimator import VideoSpeedEstimator
from agents.cctv_stream import stream_to_numpy
import numpy as np
import time
import cv2
import os
os.makedirs("videos", exist_ok=True)   # 確保輸出資料夾存在
import os
# 以像素為單位
# poly = np.array([
#     [540, 480],  # 左下
#     [853, 480],  # 右下
#     [700, 180],  # 右上
#     [400, 130],  # 左上
# ], np.float32)

w, h = 320, 240
roi_w, roi_h = int(w * 0.2), int(h * 0.2)

# 計算 ROI 左上角座標
x1 = (w - roi_w) // 2
y1 = (h - roi_h) // 2

# 計算 ROI 右下角座標
x2 = x1 + roi_w
y2 = y1 + roi_h

# 定義 ROI 的四邊形頂點 (順時針)
poly = np.array([
    [x1-105, y2+70],  # 左下
    [x2, y2+70],  # 右下
    [x2+10, y1-20],  # 右上
    [x1, y1],  # 左上
], np.float32)

estimator = VideoSpeedEstimator(
    source_video=None,
    output_video="videos/live_cctv_result.mp4",
    conf_thres=0.15,
    polygon=poly,
    model_weights="yolov10s.pt",
    frame_size=(w, h)
)

# 檢查 VideoWriter 是否成功開啟
if not estimator.writer.isOpened():
    raise RuntimeError("❌ VideoWriter failed to open – check codec or path")

# 讀串流並處理
STREAM_URL = "https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=13020"

print("🚦開始讀取串流…")

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# fps 可改
# 如果車速太快或太慢，_draw_speed_median 可以修改初始速度
# 這裡的 EMA 也可以調整使用，已經不是 EMA 是我亂調的 哈哈哈
for frame, latency in stream_to_numpy(STREAM_URL, width=w, height=h, fps=25):
    print("1. 新影像已抓取")
    start = time.time()
    processed = estimator.run(frame)
    print(f"2. 推論耗時：{time.time() - start:.2f}s")
    # processed = estimator.run(frame)   # <-- Duplicate, removed
    print(f"3. latency: {latency:.3f}s per frame\n")
    cv2.imshow("Live Detection", processed)
    if cv2.waitKey(1) == 27:  # ESC 鍵離開
        break

if estimator.writer is not None:
    estimator.writer.release()
    print(f"✅ Video saved to {estimator.out_path}")
else:
    print("⚠️ VideoWriter was not initialized.")
cv2.destroyAllWindows()
