import cv2
import time


class VideoFrameSampler:
    def __init__(self, src=0, process_fps=2):
        self.cap = cv2.VideoCapture(src)
        self.process_interval = 1.0 / process_fps
        self.last_time = 0.0

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        now = time.time()
        if now - self.last_time >= self.process_interval:
            self.last_time = now
            return frame

        return None

    def release(self):
        self.cap.release()
