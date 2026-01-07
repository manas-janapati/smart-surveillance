import json
import time
from pathlib import Path


class EdgeLogger:
    def __init__(self, log_path="data/logs/detections.jsonl"):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path

    def log(self, record):
        record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
