import cv2, time
import numpy as np
from collections import deque

class SpeedTracker:
    """每次 update 回傳 km/h 或 None"""
    
    # def __init__(self, win_sec=1.0, fps=24.0):  # highway.mp4: 測試用
    
    def __init__(self, win_sec=2.0, fps=12.0): # cctv  48 fps
        self.fps = fps
        self.win = int(fps * win_sec)
        self.hist = {}         


    # bird eye view 用像素
    def update(self, tid, bev_pt):
        t = time.time()
        x, y = bev_pt
        q = self.hist.setdefault(tid, [])
        q.append((t, x, y))

        if len(q) > self.win:
            q.pop(0)

        if len(q) < 2:
            return 0.0

        (t0, x0, y0), (t1, x1, y1) = q[0], q[-1]
        dt = t1 - t0
        if dt == 0:
            return 0.0

        dist = np.hypot(x1 - x0, y1 - y0)  # 單位是 xy, 沒轉 pixel
        v_kmh = dist / dt * 3.6              

        return v_kmh
