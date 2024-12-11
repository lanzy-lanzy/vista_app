# Real-time Vehicle Detection and Counting

This application uses YOLOv8 to detect and count vehicles in real-time using your webcam or video input.

## Features
- Real-time vehicle detection
- Vehicle counting
- Support for multiple vehicle types (cars, trucks, buses, motorcycles)
- Visual tracking with bounding boxes
- Live counter display

## Requirements
- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- NumPy
- Pillow

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Download YOLOv8 model (will be downloaded automatically on first run)

## Usage

Run the application:
```bash
python main.py
```

- Press 'q' to quit the application

## Notes
- The application uses YOLOv8n (nano) model by default
- Detection confidence threshold is set to 0.3
- The counter tracks unique vehicles using object tracking
