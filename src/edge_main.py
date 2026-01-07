import os

# -------- CPU OPTIMIZATION --------
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
torch.backends.mkldnn.enabled = True
# ---------------------------------

import cv2
from PIL import Image
from datetime import datetime, timedelta

from model_loader import load_model
from tiling import tile_image
from preprocessing import preprocess_tiles
from inference import predict_image

from video_capture import VideoFrameSampler
from event_logic import EventDetector
from edge_logger import EdgeLogger
from gps_reader import GPSReader


# -------- PATHS --------
VIDEO_PATH = "data/videos/highway_drive_multi.mp4"
GPS_PATH = "data/gps/gps_log.csv"

BACKBONE_PATH = "models/model_beit_danish.pth"
HEAD_PATH = "models/model_tiling_state_dict_xp5.pth"
# -----------------------

THRESHOLD = 0.5
TILE_BATCH_SIZE = 8


def main():
    device = "cpu"

    # Load model
    model = load_model(BACKBONE_PATH, HEAD_PATH, device)

    # Video sampler (file input)
    sampler = VideoFrameSampler(
        src=VIDEO_PATH,
        process_fps=2
    )

    # GPS reader
    gps_reader = GPSReader(GPS_PATH)

    # Video metadata
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # IMPORTANT:
    # This must be the real start time of the video recording
    video_start_time = datetime.fromisoformat(
        "2026-01-06T13:57:50+00:00"
    )

    event_detector = EventDetector(window_seconds=5, min_hits=2)
    logger = EdgeLogger()

    frame_id = 0

    print("Edge surveillance started (video + GPS).")

    while True:
        frame = sampler.read()
        if frame is None:
            break  # End of video

        frame_id += 1

        # Frame timestamp
        frame_time = video_start_time + timedelta(
            seconds=frame_id / fps
        )

        # OpenCV → PIL
        image = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        # Downscale gate
        if image.width > 768:
            new_h = int(768 * image.height / image.width)
            image = image.resize((768, new_h))

        tiles = tile_image(image)
        if not tiles:
            continue

        tile_tensor = preprocess_tiles(tiles)

        detected, confidence = predict_image(
            model,
            tile_tensor,
            threshold=THRESHOLD,
            batch_size=TILE_BATCH_SIZE
        )

        event = event_detector.update(detected)

        if event:
            lat, lon = gps_reader.get_closest(frame_time)

            record = {
                "frame_id": frame_id,
                "is_invasive": True,
                "confidence": round(confidence, 3),
                "timestamp": frame_time.isoformat().replace("+00:00", "Z"),
                "gps": {
                    "lat": lat,
                    "lon": lon
                },
                "source": os.path.basename(VIDEO_PATH)
            }

            logger.log(record)
            print(
                f"[EVENT] invasive | "
                f"{lat:.6f},{lon:.6f} | "
                f"conf={confidence:.3f}"
            )

    sampler.release()
    print("Video processing completed.")


if __name__ == "__main__":
    main()
