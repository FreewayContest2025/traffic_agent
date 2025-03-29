import json
from langchain_openai import ChatOpenAI

class officer_agent:
    def __init__(self, model = "gpt-3.5-turbo", temperature = 0):
        self.llm = ChatOpenAI(
            # 別戳太大力 orz 
            openai_api_key = "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",
            model = model,
            temperature = temperature
        )
        self.default_prompt = """
        你是一位高速公路上監控車流是否異常的警官，給定的車輛數量、車速、是否事故等資訊，需判斷是否屬於異常(壅塞/事故)或正常。
        異常例子： speed > 140, car_count > 40, accident = ture
        
        回傳JSON格式:
        {
        "status": "normal" 或 "anomaly",
        "reason": "簡短說明原因為何"
        }
        請只輸出 JSON，不要多餘文字。
        例如:
        - speed < 30, 或 car_count > 40, 或 accident = true => anomaly
        - 否則 => normal
        """

    def judge_situation(self, yolo_result: dict) -> dict:
        """"
        假設 yolo_result = 
        {
            "car_count": 10,
            "speed": 60,
            "accident": false
        }
        """
        # core function!!
        combined_prompt = f"{self.default_prompt}\n 目前道路資訊: {yolo_result}\n 請輸出 JSON"
        response = self.llm.invoke(combined_prompt)
        content = response.content.strip()
        response = json.loads(content)
        return response
    
        # 判斷 status, reason?
    