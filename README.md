# GaitVision

**GaitVision — Gait Feature Extraction & Object Detection Tool**

---

## Overview

GaitVision is a Python-based tool designed to extract gait-related features and perform object detection from videos, aimed at preprocessing data for machine learning applications. This project was developed as part of a technical assessment for Haig’s Quality Printing (Neuramate Project).

The tool processes videos (e.g., POV action scenes) to extract key metrics such as arm reach, cadence, objects present, and area size. It outputs relevant data to CSV files, suitable for further ML preprocessing.

---

## Features

- **Video Download:** Download videos from YouTube using `yt-dlp`
- **Video Playback:** Play and review videos using OpenCV
- **Gait Feature Extraction:** Extract arm reach, cadence, and area size using MediaPipe and OpenCV
- **Object Detection:** Detect common objects (person, box, door, etc.) frame-by-frame using YOLOv8 (ultralytics)
- **CSV Output:** Export extracted features and detections to CSV files for easy ML integration

---

## Setup Instructions

1. **Clone the repository**

    ```bash
    git clone https://github.com/yourusername/GaitVision.git
    cd GaitVision
    ```

2. **Install dependencies**

    Ensure you have Python 3.8+ installed.

    Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

    The key packages include:

    - `yt-dlp` — video downloading
    - `opencv-python` and `opencv-contrib-python` — video processing
    - `mediapipe` — pose estimation for gait features
    - `pandas` — data handling
    - `ultralytics` — YOLOv8 object detection

---

## Usage

- **Download a video from YouTube**

    ```bash
    python3 download_video.py
    ```

    This downloads the video from the configured URL to `gait_video.mp4`.

- **Play the video**

    ```bash
    python3 play_video.py
    ```

    This opens a window to view the video. Press `q` to exit.

- **Extract gait features**

    ```bash
    python3 extract_gait_features.py
    ```

    Processes `gait_video.mp4` to extract gait metrics and outputs a CSV file (`gait_features.csv`).

- **Perform object detection**

    ```bash
    python3 object_detection.py
    ```

    Runs YOLOv8 detection on `gait_video.mp4`, displays bounding boxes live, and outputs `object_detections.csv` containing frame-by-frame detected objects.

    Press `q` to stop the playback early.

---

## Output Files

- `gait_features.csv` — Contains extracted gait metrics per frame (arm reach, cadence, area size)
- `object_detections.csv` — Contains detected objects with bounding box coordinates per frame

---

## References

- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html) — for pose landmark detection
- [OpenCV](https://opencv.org/) — for video processing
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — for video downloading
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — for object detection

---

## Notes

- This project is designed as a proof-of-concept pipeline for gait analysis and object detection in videos.
- The choice of objects detected is based on the pre-trained YOLOv8 model and may require fine-tuning or custom training for specialized classes such as weapons or magazines.
- Performance depends on your machine’s capabilities.
