import cv2, time, logging, numpy as np
import json
from datetime import datetime

from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from speed.speed_tracker import SpeedTracker        
from collections import deque

logging.getLogger().setLevel(logging.ERROR)


class VideoSpeedEstimator:
    def __init__(
        self,
        source_video: str | None,
        output_video: str = "output.mp4",
        src_pts: np.ndarray | None = None,   
        dst_pts: np.ndarray | None = None,   
        class_id: int | None = None,
        blur_id: int | None = None,
        polygon: np.ndarray | None = None,
        # model and vehicle labels
        model_weights: str = "yolov10s.pt",
        conf_thres: float = 0.6,
        names_file: str = "category.names",
        frame_size: tuple[int, int] | None = None,
        show_track_boxes: bool = False
    ):
        self.out_path = Path(output_video)
        # 統計 JSON 輸出檔路徑
        self.stats_path = self.out_path.with_suffix(".json")
        self.conf = conf_thres
        self.class_id = class_id
        self.small_vehicle_ids = {2, 3}
        self.large_vehicle_ids = {4, 5} #truck, bus
        self.blur_id = blur_id
        self.show_track_boxes = show_track_boxes  # control whether track boxes/labels are drawn
        print("⚙️ stats_path =", self.stats_path.resolve())
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        # --- Handle live‑stream mode (no source_video) -----------------
        if source_video is None:
            self.cap = None
            self.w, self.h = frame_size if frame_size else (0, 0)
            self.fps = 5
            self.frame_size = frame_size if frame_size else (self.w, self.h)  # default FPS for live stream
        else:
            self.src_path = Path(source_video)
            self.cap = cv2.VideoCapture(str(self.src_path))
            if not self.cap.isOpened():
                raise FileNotFoundError(f"Cannot open {self.src_path}")
            # get video width, height, fps
            self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_size = (self.w, self.h)
        self.out_path = Path(output_video)

        # ROI 多邊形
        self.poly = polygon if polygon is not None else np.array(
            [[0, self.h], [self.w, self.h], [self.w, 0], [0, 0]], np.float32
        )

        # 不相關的地方在背景轉黑色 
        self.mask = np.zeros((self.h, self.w), np.uint8)
        cv2.fillPoly(self.mask, [self.poly.astype(np.int32)], 255)

        if src_pts is None or dst_pts is None:
            src_pts = self.poly.astype(np.float32)
            # 依像素調的框，目前對 road.mp4 來說算好
            dst_pts = np.array([[0, 0],
                                [450, 0],
                                [450, 350],
                                [0, 350]], np.float32)

        # 比例轉換：斜車道轉成一個平面
        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # DeepSORT: 原理尚不清: 標籤 + 追蹤
        self.tracker = DeepSort(max_age=50, n_init=1)   
        self.yolo = YOLO(model_weights)
        # 把低像素圖放大再抓
        self.yolo.overrides['imgsz'] = 320
        self.names = [n.strip() for n in open(names_file)]
        rng = np.random.default_rng(42)
        self.colors = rng.integers(0, 255, (len(self.names), 3))
        self.speed_log = deque()

        # -------- Occupancy 計數器 (60‑秒滑動視窗) --------
        self.occ_frames = 0        # 60 秒內偵測到車的影格
        self.total_frames = 0      # 60 秒內總影格
        self.last_occ_reset = time.time()

        # -------- 1‑minute 車流量 (vehicles per second) --------
        # window_max_vehicle_count : 目前 60‑秒視窗內偵測到的「最大同時車輛數」
        # last_one_minute_flow     : 上一視窗依最大車數推算的每秒車流量
        self.window_max_vehicle_count = 0
        self.last_vol_reset = time.time()     # 視窗起始時間
        self.last_one_minute_flow = 0.0

        # 速度追蹤器
        fps = self.fps
        self.speed_trk = SpeedTracker(win_sec=4.0, fps=fps)

        # VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.out_path), fourcc, self.fps, self.frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"❌ Failed to open VideoWriter with path: {self.out_path}. Check codec, frame size, or file permissions.")

    def run(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single RGB/BGR frame coming from a live stream.

        * Draw ROI polygon
        * Run YOLO → DeepSORT for detection & tracking
        * Draw per‑vehicle speed boxes, density / speed stats
        * Write to self.writer for later saving
        * Return the annotated frame for real‑time display
        """
        if frame is None or frame.size == 0:
            return frame

        if not frame.flags.writeable:
            frame = frame.copy()
        # ROI outline
        cv2.polylines(frame, [self.poly.astype(np.int32)],
                      True, (255, 0, 255), 3)

        # Detection + tracking
        tracks = self._infer_and_track(frame)

        # ---- Occupancy frame counter ------------------------------------
        # 一條「偵測線」被車輛佔據的時間比例；只有當 YOLO *本幀*
        # 真的偵測到車輛 (time_since_update == 0) 才算「有車」。
        self.total_frames += 1
        has_vehicle = any(
            trk.is_confirmed() and getattr(trk, "time_since_update", 0) == 0
            for trk in tracks
        )
        if has_vehicle:
            self.occ_frames += 1

        # Draw each confirmed track
        for trk in tracks:
            # print(trk.is_confirmed())
            if not trk.is_confirmed():
                continue
            # print(len(tracks))
            self._draw_track(frame, trk)

        # Density / speed overlay
        self._draw_density(frame, tracks)

        # Ensure frame size matches writer's expected size
        if frame.shape[1::-1] != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)

        # Save into output video
        self.writer.write(frame)
        # --------------------------------------------------
        # Expose latest_speed & latest_vehicle_count
        # --------------------------------------------------
        if self.speed_log:
            last_entry = self.speed_log[-1]
            # Case 1: stored as (timestamp, speed)
            if isinstance(last_entry, tuple) and len(last_entry) == 2:
                self.latest_speed = last_entry[1]
            # Case 2: stored as dict
            elif isinstance(last_entry, dict):
                self.latest_speed = last_entry.get("Vehicle_Median_Speed")
            else:
                self.latest_speed = None
        else:
            self.latest_speed = None

        # Count currently tracked vehicles
        try:
            self.latest_vehicle_count = len(self.speed_trk)
        except Exception:
            self.latest_vehicle_count = None
        return frame


    def _draw_speed_median(self, interval=60):
        now = time.time()
        # 舊資料超過 60 秒則刪掉，只保留前 20 %
        if self.speed_log and now - self.speed_log[0][0] > interval:
            # 最新20%
            keep = max(1, int(len(self.speed_log) * 0.2))   
            latest_entries = list(self.speed_log)[-keep:]
            self.speed_log = deque(latest_entries)
        # 重整的話回傳 60 km/h 當保底
        if not self.speed_log:
            return 60.0
        
        # 時間觀察
        # print("Oldest entry time:", datetime.fromtimestamp(self.speed_log[0][0]).strftime('%Y-%m-%d %H:%M:%S'))
        # print("Current time:", datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S'))
        # print("Time difference (seconds):", round(now - self.speed_log[0][0], 2))

        # 以目前資料計算中位數
        speeds = []
        for _, v in self.speed_log:
            # 都加上 35 ，先設好的推論誤差
            speeds.append(v + 35)
        med_speed = np.median(speeds)

        # 40 = 中位速度
        # 調整數值，可以關掉EMA
        alpha = 0.8
        med_speed = alpha * med_speed + 40
        
        return np.round(med_speed, 2)
    
    def _draw_density(self, frame, tracks, interval_m=50, max_range_m=400):
        count   = 0         
        s_count = 0
        l_count = 0

        for trk in tracks:
            if not trk.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, trk.to_ltrb())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if not (0 <= cx < self.w and 0 <= cy < self.h):
                continue
            if self.mask[cy,cx] == 0:
                continue
            _, my = self._pixel_to_bev((cx,cy))
            id = trk.get_det_class()
            # 總車輛計算還有區分
            if 0 <= my <= max_range_m:
                count += 1
                if id in self.small_vehicle_ids:
                    s_count += 1
                else:
                    l_count += 1

        dens = count / (max_range_m / 100)
        
        # Occupancy: 10 秒裡有 ？% 的時間，偵測框內都看得到車。
        if time.time() - self.last_occ_reset >= 60:
            occupancy = round(self.occ_frames / max(1, self.total_frames) * 100, 2)
            # reset window
            self.occ_frames = 0
            self.total_frames = 0
            self.last_occ_reset = time.time()
        else:
            occupancy = round(self.occ_frames / max(1, self.total_frames) * 100, 2)

        med_speed = self._draw_speed_median()

        # ---- 1‑minute flow (vehicles per second) 計算 ----
        # 更新目前視窗內的最大同時車輛數
        self.window_max_vehicle_count = max(self.window_max_vehicle_count, count)

        # 若視窗已滿 60 秒，計算上一視窗統計並重置
        if time.time() - self.last_vol_reset >= 60:
            # 以上一視窗的最大同時車輛數估算每秒流量
            self.last_one_minute_flow = round(self.window_max_vehicle_count / 60, 4)
            # 重置視窗
            self.window_max_vehicle_count = count  # 以當前幀作為新視窗初值
            self.last_vol_reset = time.time()

        one_minute_flow = self.last_one_minute_flow
        one_minute_max_count = self.window_max_vehicle_count

        text1 = f"Small vehicles count: {s_count}"
        (text_w1, text_h1), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x1 = 352 - text_w1 + 50
        y1 = 240 - 10

        text2 = f"Large vehicles count: {l_count}"
        (_, text_h2), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x2 = x1
        y2 = y1 - text_h2 - 5

        text3 = f"Density: {dens:.4f} veh/100m"
        (_, text_h3), _ = cv2.getTextSize(text3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x3 = x1
        y3 = y2 - text_h3 - 5

        text4 = f"Speed: {med_speed} kms/hr"
        (_, text_h4), _ = cv2.getTextSize(text4, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x4 = x1
        y4 = y3 - text_h4 - 5


        #計算小車數量
        cv2.putText(frame, text1, (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        #計算大車數量
        cv2.putText(frame, text2, (x2, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        #計算密度
        cv2.putText(frame, text3, (x3, y3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        #計算中位速度
        cv2.putText(frame, text4, (x4, y4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # 以 JSONL 方式累積寫入統計資料（每行一筆）
        stats = {
            "TimeStamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Occupancy": occupancy,
            "Vehicle_Median_Speed": np.round(med_speed, 2) if med_speed is not None else None,
            "VehicleType_S_Volume": s_count,
            "VehicleType_L_Volume": l_count,
            "1_minute_flow": one_minute_flow,
            "Density": round(dens, 4),
            "one_minute_max_count": one_minute_max_count
        }
        with open(self.stats_path, "a") as f:        # 追加寫入
            f.write(json.dumps(stats) + ",\n")        # 每筆獨立一行


    def _infer_and_track(self, frame):
        dets = []
        for p in self.yolo(frame, verbose=False):
            for b in p.boxes:
                # 標 yolo 抓到的物件框
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                # cid: object 類別, conf: 信心值
                cid, conf = int(b.cls[0]), b.conf[0]
                # 信心值太低就跳過
                if conf < self.conf:
                    continue
                # 可以設定特定類別
                if cid not in self.small_vehicle_ids and cid not in self.large_vehicle_ids:
                    continue
                # 抓物件的中心位置，如果在 ROI(框框) 外就跳過（前面有設為0了）
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if self.mask[cy, cx] == 0:
                    continue
                dets.append([[x1, y1, x2 - x1, y2 - y1], conf, cid])

        return self.tracker.update_tracks(dets, frame=frame)


    # 加了會不準，但好像才是對的
    def _pixel_to_bev(self, pt):
        p = np.array([[pt]], dtype=np.float32)
        q = cv2.perspectiveTransform(p, self.M)[0][0]
        return q  # 公尺單位


    def _draw_track(self, frame, trk, scale=1):
        tid = trk.track_id
        
        x1, y1, x2, y2 = map(int, trk.to_ltrb())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if not (0 <= cx < self.w and 0 <= cy < self.h):
            return
        if self.mask[cy, cx] == 0:
            return
        mx, my = self._pixel_to_bev((cx, cy))  # bottom center 點轉成 BEV
        v = self.speed_trk.update(tid, (mx, my))  # 丟給 SpeedTracker
        if v > 0:
            self.speed_log.append((time.time(), v)) 
        # skip drawing if visual boxes are turned off
        if not self.show_track_boxes:
            return
        spd_txt = f"{v:.1f} km/h" if v else ""

        cid = trk.get_det_class()

        if cid in self.small_vehicle_ids:
            category = "Small"
        elif cid in self.large_vehicle_ids:
            category = "Large"
        else:
            return
        
        B, G, R = map(int, self.colors[cid])
        
        # 框 + 四角
        self._draw_corner_rect(frame, (x1, y1, x2 - x1, y2 - y1),
                               (B, G, R), (R, G, B),
                               line_len=25, line_thk=5, rect_thk=2)

        #上方標籤
        # id_txt = f"{tid} - {category}"
        # cv2.rectangle(frame, (x1, y1 - 28), (x1 + len(id_txt) * 12, y1 - 4),
        #               (B, G, R), -1)
        # cv2.putText(frame, id_txt, (x1 + 4, y1 - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 下方速度標籤
        if spd_txt:
            cv2.rectangle(frame, (x1, y2 + 4), (x1 + len(spd_txt) * 12, y2 + 28),
                          (0, 0, 255), -1)
            cv2.putText(frame, spd_txt, (x1 + 4, y2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 打碼: 想打碼的車種可以放
        # if self.blur_id is not None and cid == self.blur_id:
        #     frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)


    # 畫框
    @staticmethod
    def _draw_corner_rect(img, bbox, rect_color, line_color,
                          line_len=25, line_thk=5, rect_thk=2):
        x, y, w, h = map(int, bbox)
        x1, y1 = x + w, y + h
        cv2.rectangle(img, (x, y), (x1, y1), rect_color, rect_thk)
        pts = [((x, y), (x + line_len, y)),
               ((x, y), (x, y + line_len)),
               ((x1, y), (x1 - line_len, y)),
               ((x1, y), (x1, y + line_len)),
               ((x, y1), (x + line_len, y1)),
               ((x, y1), (x, y1 - line_len)),
               ((x1, y1), (x1 - line_len, y1)),
               ((x1, y1), (x1, y1 - line_len))]
        for p1, p2 in pts:
            cv2.line(img, p1, p2, line_color, line_thk)


    def _release(self):
        if self.cap is not None:
            self.cap.release()
        self.writer.release()
