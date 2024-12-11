import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# Initialize video capture (0 for webcam, or provide video file path)
cap = cv2.VideoCapture(0)

# Store the track history
track_history = defaultdict(lambda: [])

# Initialize counter
vehicle_count = 0
counted_ids = set()

# Colors for visualization
COLORS = {
    'box': (0, 255, 0),  # Green
    'text': (255, 255, 255),  # White
    'count_bg': (0, 0, 0),  # Black
    'count_text': (255, 255, 255)  # White
}

def draw_boxes(frame, detections):
    global vehicle_count, counted_ids
    
    # Process each detection
    for detection in detections:
        boxes = detection.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Filter for vehicles (car, truck, bus, motorcycle)
            if cls in [2, 5, 7, 3] and conf > 0.3:
                # Draw detection box
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS['box'], 2)
                
                # Add tracking ID to counted set if not already counted
                if hasattr(box, 'id'):
                    track_id = int(box.id[0])
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        vehicle_count += 1
                
                # Draw label
                label = f"{model.names[cls]} {conf:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), COLORS['box'], -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 2)

    # Draw vehicle count
    count_text = f"Vehicles Detected: {vehicle_count}"
    cv2.rectangle(frame, (10, 10), (250, 50), COLORS['count_bg'], -1)
    cv2.putText(frame, count_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['count_text'], 2)
    
    return frame

def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 tracking on the frame
        results = model.track(frame, persist=True, verbose=False)

        # Draw detections
        frame = draw_boxes(frame, results)

        # Display the frame
        cv2.imshow('Vehicle Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
