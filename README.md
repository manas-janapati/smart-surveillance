# Edge-Based Invasive Plant Surveillance System

This project implements an **edge-based computer vision pipeline** for detecting **invasive plant species** along highways using video footage.
The system is designed to operate on **CPU-only edge devices**, perform **real-time or near–real-time inference**, and generate **geotagged detection logs** suitable for downstream GIS systems or APIs.

---

## 1. Project Overview

The system processes video footage captured from a vehicle-mounted or roadside camera, converts the video into frames, applies a **BEiT-based deep learning model with tiling**, detects invasive plant presence, and records **geotagged detection events** in JSON format.

### Key Characteristics

* Edge-first (CPU-only, no GPU required)
* Real-time capable (configurable FPS)
* Offline replay supported (MP4 input)
* Geotagged detections using GPS logs
* Modular, extensible architecture

---

## 2. System Architecture

```
Video Source (Camera / MP4)
        ↓
Frame Sampling (FPS Control)
        ↓
Image Tiling
        ↓
BEiT Backbone + Classifier Head
        ↓
Tile Aggregation
        ↓
Event Logic (Temporal Filtering)
        ↓
Geotagging (GPS Sync)
        ↓
JSON Logs / API Output
```

---

## 3. Repository Structure

```
invasive_surveillance/
│
├── models/
│   ├── model_beit_danish.pth
│   └── model_tiling_state_dict_xp5.pth
│
├── data/
│   ├── videos/
│   │   └── highway_drive_multi.mp4
│   └── gps/
│       └── gps_log.csv
│
├── src/
│   ├── edge_main.py          # Main edge pipeline
│   ├── model_loader.py       # Loads backbone + head
│   ├── preprocessing.py     # Image normalization
│   ├── tiling.py             # Image tiling logic
│   ├── inference.py          # Batched inference
│   ├── video_capture.py      # FPS-controlled video reader
│   ├── event_logic.py        # Temporal event detection
│   ├── edge_logger.py        # JSONL logger
│   └── gps_reader.py         # GPS timestamp synchronization
│
├── requirements.txt
└── README.md
```

---

## 4. Models Used

* **Backbone**: BEiT (pretrained on ImageNet, fine-tuned on Pl@ntNet)
* **Classifier Head**: Tiling-based invasive species head (XP5)
* **Classes**: 6 invasive plant species
* **Input Size**: 384×384 (tiling), downscaled from video frames

---

## 5. Video Input Modes

The pipeline supports multiple video sources:

| Mode        | Source              |
| ----------- | ------------------- |
| Development | Webcam (`src=0`)    |
| Validation  | Video file (`.mp4`) |
| Deployment  | RTSP / IP camera    |

Example (video file mode):

```python
sampler = VideoFrameSampler(
    src="data/videos/highway_drive_multi.mp4",
    process_fps=2
)
```

---

## 6. GPS Geotagging (Development Mode)

### Important Note

Video files do **not** contain GPS data.
Geotagging is performed by **time-aligning frames with an external GPS log**.

### GPS Log Format (`gps_log.csv`)

```csv
timestamp,lat,lon
2026-01-06T13:57:50Z,55.870820,9.464190
2026-01-06T13:57:51Z,55.870824,9.464198
...
```

* ISO-8601 timestamps (UTC)
* Typically recorded at 1 Hz or higher
* Synchronized with video start time

---

## 7. Output Format

Detections are written as **JSON Lines (`.jsonl`)**, one record per event.

Example:

```json
{
  "frame_id": 15,
  "is_invasive": true,
  "confidence": 0.512,
  "timestamp": "2026-01-07T05:22:06Z",
  "gps": {
    "lat": 55.937710,
    "lon": 9.716395
  },
  "source": "highway_drive_multi.mp4"
}
```

This format is:

* GIS-compatible
* API-ready
* Auditable
* Replayable

---

## 8. Performance Characteristics

Typical CPU-only performance (laptop-class hardware):

| Parameter              | Value        |
| ---------------------- | ------------ |
| Processing FPS         | ~1–2 FPS     |
| Capture FPS            | Configurable |
| Detection latency      | ~1–2 seconds |
| Vehicle speed (survey) | 50–80 km/h   |

Inference speed does **not** constrain vehicle speed when processing offline.

---

## 9. How to Run

### 1. Create environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2. Run edge pipeline

```bash
python src/edge_main.py
```

---

## 10. Current Limitations (By Design)

* CPU-only inference (no GPU acceleration)
* GPS interpolation uses nearest-neighbor (development stage)
* No real-time map visualization yet
* No REST API exposed yet

---

## 11. Planned Extensions

* GPS interpolation between points
* GeoJSON export for GIS tools
* REST API for detections
* ONNX Runtime optimization
* Multi-camera support
* Frame saving on detection
* Real-time dashboard

---

## 12. Intended Use

This system is designed for:

* Highway authorities
* Environmental monitoring agencies
* Roadside vegetation surveys
* Research and pilot deployments

It is **not** intended for consumer-grade mobile devices.

---

## 13. License

This project follows the license terms of the underlying models (CC-BY 4.0 where applicable).
Refer to the original model publication for attribution requirements.

---

## 14. Citation

If you use this work in research or pilots, please cite the original model source:

Espitalier et al. (2024). *Image classification models dedicated to the identification of invasive plant species in roadside views*.

---

## 15. Status

**Active development – Edge prototype phase**
