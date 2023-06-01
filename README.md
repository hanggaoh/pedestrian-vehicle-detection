# Real-Time Pedestrian and Vehicle Detection

This project is a real-time object detection system designed for embedded systems like Raspberry Pi and Jetson Nano. It uses OpenCV and YOLOv5 to detect pedestrians and vehicles from a live camera feed or video file.

## Features

*   Real-time pedestrian and vehicle detection.
*   Powered by YOLOv5 for high accuracy.
*   Optimized for low-latency performance on embedded hardware (with CUDA acceleration support).
*   Supports both live camera feeds and video files.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/pedestrian-vehicle-detection.git
    cd pedestrian-vehicle-detection
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download YOLOv5 weights:**
    You need to download the YOLOv5 pre-trained weights.
    ```bash
    mkdir weights
    wget -P weights https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
    ```
    This project is configured to use `yolov5s.pt` by default, which is a small and fast model suitable for embedded systems.

## Usage

To run the detection on a video file:
```bash
python main.py --source path/to/your/video.mp4
```

To run the detection on a live camera feed (e.g., camera index 0):
```bash
python main.py --source 0
```
