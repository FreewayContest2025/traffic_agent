"""
LLM 偵測
"""

import time
from agents.cctv_stream import camera_agent
from agents.yolo_agent import yolo_agent
from agents.observer_agent import officer_agent
from agents.alertion_agent import alertion_agent

class traffic_agent:
    def __init__(self):
        self.camera_agent = camera_agent()
        self.yolo_agent = yolo_agent()
        self.officer_agent = officer_agent()
        self.alertion_agent = alertion_agent()

    def main(self):
        
        # while True: 開始監控
        # camera_agent.get_frame()
        # ...
        # analyed_result = yolo_agent.analyze_frame(frame)
        
        # this result is for testing
        analyzed_result = { "car_count": 100, "speed": 20, "accident": False }

        judge_result = self.officer_agent.judge_situation(analyzed_result)
        if judge_result["status"] == "anomaly":
            self.alertion_agent.handle_alert(judge_result)
            print(judge_result)
        else:
            print(judge_result)
            print("nothing happened")

if __name__ == "__main__":
    traffic_agent = traffic_agent()
    traffic_agent.main()