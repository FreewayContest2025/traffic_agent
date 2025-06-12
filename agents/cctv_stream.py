# 這裡要寫怎麼處理 CCTV 的圖片
import subprocess as sp
import numpy as np
import shlex 
import time
import cv2
import os 

def stream_to_numpy(url: str,  width: int = 1280, height: int = 720, fps: int = 5):
    ffmpeg_cmd = (
        f"ffmpeg -loglevel error "
        f"-i {shlex.quote(url)} "
        f"-vf scale={width}:{height},fps={fps} "
        "-f image2pipe -pix_fmt bgr24 -vcodec rawvideo -"
    )
    proc = sp.Popen(shlex.split(ffmpeg_cmd), stdout=sp.PIPE, bufsize=height*width*3*2)
    frame_len = width * height * 3  # BGR24
    try:
        while True:
            start = time.time()
            raw = proc.stdout.read(frame_len)
            if len(raw) < frame_len:      # 串流中斷
                print("⚠️  Stream ended or cannot fetch frame")
                break
            frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))
            yield frame, (time.time() - start)  # 回傳影格與抓取耗時
    finally:
        proc.terminate()

STREAM_URL = "https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=40430"
RECONNECT_S = 2
FRAME_DIR = "frames"
VIDEO_DIR = "videos"
FRAME_SAVE_INTERVAL = 2 # 每 2 frames拍一次，+-1.?秒(latency)，約是 0~2 秒拍一次
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

saved_frames = []
width, height, fps = 1280, 720, 5
FRAMES_PER_VIDEO = fps * 4 # 20 個 frames 一個影片
video_counter = 0
frame_counter = 0

while True:
    for img, latency in stream_to_numpy(STREAM_URL):
        print(f"[{frame_counter}] Frame latency: {latency:.3f}s")
        # 每 10 frames 存一次
        frame_counter += 1
        if frame_counter % FRAME_SAVE_INTERVAL == 0:
            filename = os.path.join(FRAME_DIR, f"frame_{frame_counter:06d}.jpg")
            cv2.imwrite(filename, img)
            saved_frames.append(filename)
            print(f"Saved {filename}")
        cv2.imshow("Live", img)
        if cv2.waitKey(1) == 27:  # ESC 離開
            break
        print("Stream unconnected, it will be reconnected later.")
        # 每 10 張存成一個影片
        if len(saved_frames) >= FRAMES_PER_VIDEO:
            print("Combining frames into video output.mp4 ...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_path = os.path.join(VIDEO_DIR, f"video_{video_counter:03d}.mp4")
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))           
            for fpath in saved_frames:
                img = cv2.imread(fpath)
                if img is not None:
                    out.write(img)
            out.release()
            print(f"Video saved as {video_path}")
            video_counter +=1
            saved_frames = []
