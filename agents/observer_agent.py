import json
from pathlib import Path
import os
import re
import typing
import openai
from dotenv import load_dotenv
class TrafficSpeedLevelAgent:

    LEVEL_TABLE = [
        {"level": 1, "name": "順暢", "low": 80, "high": None},     
        {"level": 2, "name": "車多", "low": 60, "high": 79},
        {"level": 3, "name": "壅塞", "low": 40, "high": 59},
        {"level": 4, "name": "嚴重壅塞", "low": 20, "high": 39},
        {"level": 5, "name": "極度壅塞", "low": 0,  "high": 19},
        {"level": -1, "name": "道路封閉", "low": -1, "high": -1},  
    ]

    _SYSTEM_PROMPT = """You are an intelligent freeway-traffic assistant.
    Based on the following Taiwan Freeway speed‑level definition, choose the correct
    congestion level for the incoming snapshot and briefly explain your decision.
    Return ONLY a compact JSON:
    {{
        "Level": <int>,          # –1, 1‑5
        "LevelName": <string>,   # Chinese name
        "Reason": <string>       # ≤25 Chinese characters
    }}

    Speed‑level table (km/h):
    1 順暢:          speed ≥ 80
    2 車多:     60 ≤ speed ≤ 79
    3 壅塞:     40 ≤ speed ≤ 59
    4 嚴重壅塞: 20 ≤ speed ≤ 39
    5 極度壅塞:  0 ≤ speed ≤ 19
    –1 道路封閉:   speed < 0 or sensor indicates closure

    Additional rules:
    * Use –1 only when speed is negative or an explicit closure flag appears.
    * If the previous 5‑minute history shows a worsening trend (level increasing
    by ≥1) and current speed falls within 2 km/h of a boundary, you MAY escalate
    the level by 1 and mention the trend.
    * Never output text outside the JSON braces.
    """

    # ---------------------------------------------------------
    #  Construction
    # ---------------------------------------------------------
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        *,
        timeout: typing.Optional[int] = None,
    ) -> None:
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")  # ← loads .env file in project root
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. "
                "Create a .env file with OPENAI_API_KEY=<your‑key>."
            )
        openai.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    # ---------------------------------------------------------
    #  Public API
    # ---------------------------------------------------------
    def suggest_level(
        self,
        current: dict[str, any],
        history: list[dict[str, any]] = None,
    ) -> dict[str, any]:
        """
        Call GPT and return the parsed JSON suggestion.

        Parameters
        ----------
        current : dict
            Snapshot dict; must contain 'Vehicle_Median_Speed'.
        history : list[dict] | None
            Optional list with key '5分鐘後壅塞程度' in each row.
        """
        if history is None:
            history = []

        prompt = self._build_prompt(current, history)

        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            temperature=self.temperature,
            timeout=self.timeout,
        )
        content = response.choices[0].message.content.strip()
        return self._parse_json(content)

    # ---------------------------------------------------------
    #  Helpers
    # ---------------------------------------------------------
    def _build_prompt(
        self,
        current: dict[str, any],
        history: list[dict[str, any]],
    ) -> str:
        user_block = json.dumps(current, ensure_ascii=False, indent=2)
        prompt = f"{self._SYSTEM_PROMPT}\n\nCurrent snapshot:\n```json\n{user_block}\n```\n"
        if history:
            hist_block = json.dumps(history[-20:], ensure_ascii=False, indent=2)
            prompt += f"History (latest ≤20):\n```json\n{hist_block}\n```\n"
        prompt += "Respond with the suggested level only."
        return prompt

    @staticmethod
    def _parse_json(content: str) -> dict[str, any]:
        """Extract the first JSON object from GPT output."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{[^{}]*?(?:\{[^{}]*?\}[^{}]*?)*\}", content, re.S)
            if match:
                return json.loads(match.group())
            raise ValueError(f"GPT returned non‑JSON content:\n{content}")

    # ---------------------------------------------------------
    #  CLI helper
    # ---------------------------------------------------------
    def _cli(self) -> None:  # pragma: no cover
        import sys
        if sys.stdin.isatty():
            print(
                "Usage:\n"
                "  cat snapshots.json  | python observer_agent.py   # JSON array\n"
                "  cat snapshots.jsonl | python observer_agent.py   # JSON‑Lines",
                file=sys.stderr,
            )
            return
        
        snapshots_iter = sys.stdin     # direct iterator over lines
        out_path = "./docs/suggestion.json"

        # -------- prepare output file (persistent JSON array) --------
        if not os.path.exists(out_path):
            # first‑time creation
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("[\n")
        else:
            # safely reopen file, remove trailing ']' and possible newline, then add comma
            with open(out_path, "r+", encoding="utf-8") as f:
                content = f.read().rstrip()
                if content.endswith("]"):
                    content = content[:-1].rstrip()  # drop ']'
                if content and not content.endswith(","):
                    content += ","
                content += "\n"
                f.seek(0)
                f.write(content)
                f.truncate()

        processed = 0
        for idx, line in enumerate(snapshots_iter, 1):
            line = line.strip()

            # skip empty lines and array brackets
            if not line or line in ("[", "]"):
                continue

            # drop optional trailing comma
            if line.endswith(","):
                line = line[:-1].rstrip()

            try:
                snap = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {idx}: invalid JSON – {e}", file=sys.stderr)
                continue

            if idx % 10 == 1:
                print(f"[{idx}] Processing...", file=sys.stderr, flush=True)

            try:
                result = self.suggest_level(snap, [])
                with open(out_path, "a", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False)
                    f.write(",\n")
                processed += 1
            except Exception as e:
                print(f"Snapshot error: {e}", file=sys.stderr)

        # finalise: replace last ",\n" with "\n]\n"
        with open(out_path, "rb+") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                # nothing written – create empty array
                f.write(b"]\n")
            else:
                # move back two bytes and read to find the last comma/newline pattern
                step = min(5, size)
                f.seek(-step, os.SEEK_END)
                tail = f.read(step)
                # find last comma
                last_comma = tail.rfind(b",")
                if last_comma != -1:
                    f.seek(-step + last_comma, os.SEEK_END)
                    f.truncate()
                f.write(b"\n]\n")

"""
use this command to activate
cat videos/cctv_6_16_19.json | python3 agents/observer_agent.py
"""

if __name__ == "__main__":
    TrafficSpeedLevelAgent()._cli()