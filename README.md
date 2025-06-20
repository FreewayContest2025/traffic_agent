
1. build env 
```
pip install -r requirements.txt
```
2. server
```
python3 main1.py 
```
3. client
 ```
curl -N http://localhost:8000/api/traffic/stream\?camera_id\=13020
 ```

4. LLM，記得配 api key

```
cat videos/live_cctv_result.json | python3 agents/observer_agent.py
```

![architecture](./docs/architecture.png)