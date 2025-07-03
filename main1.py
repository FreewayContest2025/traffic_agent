"""
抓速度、密度、大車、小車
"""

from speed.speed_estimator import VideoSpeedEstimator
from agents.cctv_stream import stream_to_numpy
import numpy as np
import time
import cv2
import os

# Directory containing JSON summaries; override via env TRAFFIC_JSON_DIR if needed
JSON_DIR = os.environ.get("TRAFFIC_JSON_DIR", "videos")

import json
from flask import Flask, jsonify, request, Response, stream_with_context, send_file, abort
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

def _open_writer(path: str, fps: int = 25, frame_size: tuple[int, int] = (w, h)):
    """
    Create and return a cv2.VideoWriter using a codec that plays in most players.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4/H.264‑compatible
    return cv2.VideoWriter(path, fourcc, fps, frame_size)

# ----------  RESTful service wrapper  ----------

def analyze_camera(camera_id: str, sample_frames: int = 50):
    """
    Sample a few frames from the given freeway CCTV camera, run the
    YOLOv10-based VideoSpeedEstimator, and return simple metrics as a dict.
    """
    video_path = os.path.join("videos", f"tmp_{camera_id}.mp4")
    estimator = VideoSpeedEstimator(
        source_video=None,
        output_video=video_path,  # final playable file
        conf_thres=0.15,
        polygon=poly,
        model_weights="yolov10s.pt",
        frame_size=(w, h)
    )
    if not estimator.writer.isOpened():
        raise RuntimeError("❌ VideoWriter failed to open – check codec or path")

    speeds = []
    vehicle_counts = []
    for idx, (frame, latency) in enumerate(
        # 此處可調
        stream_to_numpy(f"https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera={camera_id}", width=w, height=h, fps=25)
    ):
        _ = estimator.run(frame)

        cv2.imshow("Live Detection", frame)
        if cv2.waitKey(1) == 27:  # ESC to break
            break

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

    cv2.destroyAllWindows()

    estimator.writer.release()
    result = {
        "camera_id": camera_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "frames_sampled": len(speeds),
        "average_speed": float(np.mean(speeds)) if speeds else None,
        "average_vehicle_count": float(np.mean(vehicle_counts)) if vehicle_counts else None,
    }
    result["video_path"] = f"videos/tmp_{camera_id}.mp4"
    return result


def monitor_stream(camera_id: str, *, display: bool = False):
    """
    Generator that pulls the CCTV stream, runs VideoSpeedEstimator,
    yields newline‑delimited JSON metrics to the caller, and writes:

    * videos/tmp_<camera_id>.mp4   – recorded video (written by estimator)
    * videos/tmp_<camera_id>.json  – newline‑delimited JSON log

    When the caller closes the HTTP connection (or Ctrl‑C on curl),
    the generator is closed → finally‑block runs → resources released.
    """
    video_path = os.path.join("videos", f"tmp_{camera_id}.mp4")
    json_path  = os.path.join(JSON_DIR, f"tmp_{camera_id}.json")
    # start a fresh json log
    open(json_path, "w").close()

    estimator = VideoSpeedEstimator(
        source_video=None,
        output_video=video_path,
        conf_thres=0.15,
        polygon=poly,
        model_weights="yolov10s.pt",
        frame_size=(w, h)
    )
    if not estimator.writer.isOpened():
        raise RuntimeError("❌ VideoWriter failed to open – check codec or path")

    try:
        for frame, _ in stream_to_numpy(
            f"https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera={camera_id}",
            width=w, height=h, fps=25
        ):
            processed = estimator.run(frame)  # estimator internally writes MP4

            # optional live display (only works in main thread with GUI)
            if display:
                cv2.imshow("Live Detection", processed)
                if cv2.waitKey(1) == 27:
                    break

            # keep‑alive status line to client
            yield f"正在拍攝 {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    finally:
        if display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        estimator.writer.release()


@app.route("/api/traffic/stream", methods=["GET"])
def traffic_stream():
    """
    Stream newline‑delimited status lines to the client *and* (optionally)
    show the realtime detection window on the server side.

    Enable the GUI preview by calling, for example:
        curl -N "http://localhost:8000/api/traffic/stream?camera_id=13020&display=1"
    """
    camera_id = request.args.get("camera_id", default="13020")
    # any of 1 / true / yes enables display
    disp_flag  = request.args.get("display", default="0").lower() in {"1", "true", "yes"}

    def _stream():
        yield f"開始偵測 {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        yield from monitor_stream(camera_id, display=disp_flag)

    return Response(stream_with_context(_stream()), mimetype="text/plain")


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


# --------- New route for serving video JSON ---------
@app.route("/api/video_json", methods=["GET"])
def video_json():
    """
    Return the JSON summary file for a given camera, e.g.
        GET /api/video_json?camera_id=13020
    looks for videos/tmp_<camera_id>.json  and streams it to the client.
    If the file does not exist, responds with 404.
    """
    camera_id = request.args.get("camera_id", default="13020")
    # Allow full filename override; otherwise construct from camera_id
    filename = request.args.get("filename")
    if filename:
        json_path = filename
    else:
        json_path = os.path.join(JSON_DIR, f"tmp_{camera_id}.json")

    if not os.path.isfile(json_path):
        abort(404, description=f"No JSON found at {json_path}")

    # Read file once
    with open(json_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Try parse as a single JSON object/array
    try:
        payload = json.loads(raw)
        return jsonify(payload)
    except json.JSONDecodeError:
        # Probably newline‑delimited JSON; return raw text
        return Response(raw, mimetype="application/json")


if __name__ == "__main__":
    print("🚦 Traffic agent API running at http://0.0.0.0:8000/api/traffic")
    # analyze_camera("13020", 50)
    app.run(host="0.0.0.0", port=8000, debug=True)
