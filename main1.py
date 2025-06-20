"""
抓速度、密度、大車、小車
"""

from speed.speed_estimator import VideoSpeedEstimator
from agents.cctv_stream import stream_to_numpy
import numpy as np
import time
import cv2
import os
import json
from flask import Flask, jsonify, request, Response, stream_with_context
app = Flask(__name__)
os.makedirs("videos", exist_ok=True)   # 確保輸出資料夾存在
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

# ----------  RESTful service wrapper  ----------

def analyze_camera(camera_id: str, sample_frames: int = 50):
    """
    Sample a few frames from the given freeway CCTV camera, run the
    YOLOv10-based VideoSpeedEstimator, and return simple metrics as a dict.
    """
    STREAM_URL = f"https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera={camera_id}"
    estimator = VideoSpeedEstimator(
        source_video=None,
        output_video=f"videos/tmp_{camera_id}.mp4",  # temporary sink file
        conf_thres=0.15,
        polygon=poly,
        model_weights="yolov10s.pt",
        frame_size=(w, h)
    )

    speeds = []
    vehicle_counts = []
    for idx, (frame, latency) in enumerate(
        # 此處可調
        stream_to_numpy(STREAM_URL, width=w, height=h, fps=25)
    ):
        _ = estimator.run(frame)

        # These attributes depend on your VideoSpeedEstimator implementation.
        # Replace them with the correct property/method names if different.
        speed = getattr(estimator, "latest_speed", None)
        count = getattr(estimator, "latest_vehicle_count", None)

        if speed is not None:
            speeds.append(speed)
        if count is not None:
            vehicle_counts.append(count)

        if idx + 1 >= sample_frames:
            break

    result = {
        "camera_id": camera_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "frames_sampled": len(speeds),
        "average_speed": float(np.mean(speeds)) if speeds else None,
        "average_vehicle_count": float(np.mean(vehicle_counts)) if vehicle_counts else None,
    }
    return result


def monitor_stream(camera_id: str):
    """
    Yield newline-delimited JSON with live metrics.
    Client example:
        curl -N http://localhost:8000/api/traffic/stream?camera_id=13020
    """
    STREAM_URL = f"https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera={camera_id}"
    estimator = VideoSpeedEstimator(
        source_video=None,
        output_video=f"videos/tmp_{camera_id}.mp4",
        conf_thres=0.15,
        polygon=poly,
        model_weights="yolov10s.pt",
        frame_size=(w, h)
    )

    for frame, latency in stream_to_numpy(STREAM_URL, width=w, height=h, fps=25):
        _ = estimator.run(frame)
        speed  = getattr(estimator, "latest_speed", None)
        count  = getattr(estimator, "latest_vehicle_count", None)

        data = {
            "camera_id": camera_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            "speed": speed,
            "vehicle_count": count,
        }
        yield json.dumps(data) + "\n"


@app.route("/api/traffic/stream", methods=["GET"])
def traffic_stream():
    """
    Keep the HTTP connection open.
    Continuously show "正在拍攝 <timestamp>" every second to the client,
    while the server silently processes the CCTV stream in the background.
    Example client:
        curl -N http://localhost:8000/api/traffic/stream?camera_id=13020
    """
    camera_id = request.args.get("camera_id", default="13020")

    # --- Start background detection thread (daemon) ---
    def _bg_task():
        for _ in monitor_stream(camera_id):
            pass  # just consume to keep processing

    import threading, time as _t
    threading.Thread(target=_bg_task, daemon=True).start()

    # --- Generator that keeps client connection alive ---
    def _status_gen():
        while True:
            yield f"正在拍攝 {_t.strftime('%Y-%m-%d %H:%M:%S')}\n"
            _t.sleep(1)

    return Response(stream_with_context(_status_gen()), mimetype="text/plain")


@app.route("/api/traffic", methods=["GET"])
def traffic_endpoint():
    """
    Example:
        GET /api/traffic?camera_id=13020
    """
    camera_id = request.args.get("camera_id", default="13020")
    frames = int(request.args.get("frames", default="50"))
    data = analyze_camera(camera_id, frames)
    return jsonify(data)


if __name__ == "__main__":
    print("🚦 Traffic agent API running at http://0.0.0.0:8000/api/traffic")
    app.run(host="0.0.0.0", port=8000, debug=True)
