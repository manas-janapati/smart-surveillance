import time
from collections import deque


class EventDetector:
    def __init__(self, window_seconds=5, min_hits=2):
        self.window_seconds = window_seconds
        self.min_hits = min_hits
        self.hits = deque()

    def update(self, detected):
        now = time.time()

        if detected:
            self.hits.append(now)

        while self.hits and now - self.hits[0] > self.window_seconds:
            self.hits.popleft()

        return len(self.hits) >= self.min_hits
