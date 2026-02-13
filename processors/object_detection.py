"""
Object Detection Processor
Detects 80 object classes using YOLOv8 nano
"""

import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLOv8 nano model (smallest, fastest)
model = YOLO("yolov8n.pt")

# Industrial color palette for different object categories
CATEGORY_COLORS = {
    "person": (0, 255, 170),      # Cyan-green
    "vehicle": (255, 100, 0),     # Orange
    "animal": (0, 200, 255),      # Amber
    "furniture": (128, 100, 200), # Purple
    "electronic": (255, 200, 0),  # Yellow
    "food": (100, 200, 100),      # Green
    "default": (180, 180, 180),   # Gray
}

# Category mapping
CATEGORY_MAP = {
    "person": "person",
    "bicycle": "vehicle", "car": "vehicle", "motorcycle": "vehicle",
    "airplane": "vehicle", "bus": "vehicle", "train": "vehicle",
    "truck": "vehicle", "boat": "vehicle",
    "bird": "animal", "cat": "animal", "dog": "animal", "horse": "animal",
    "sheep": "animal", "cow": "animal", "elephant": "animal", "bear": "animal",
    "zebra": "animal", "giraffe": "animal",
    "chair": "furniture", "couch": "furniture", "bed": "furniture",
    "dining table": "furniture", "toilet": "furniture",
    "tv": "electronic", "laptop": "electronic", "mouse": "electronic",
    "remote": "electronic", "keyboard": "electronic", "cell phone": "electronic",
    "microwave": "electronic", "oven": "electronic", "toaster": "electronic",
    "refrigerator": "electronic",
    "banana": "food", "apple": "food", "sandwich": "food", "orange": "food",
    "broccoli": "food", "carrot": "food", "hot dog": "food", "pizza": "food",
    "donut": "food", "cake": "food",
}


def get_color(class_name: str) -> tuple:
    """Get color based on object category."""
    category = CATEGORY_MAP.get(class_name, "default")
    return CATEGORY_COLORS.get(category, CATEGORY_COLORS["default"])


def process(frame: np.ndarray) -> np.ndarray:
    """
    Process a frame and detect objects.
    
    Args:
        frame: BGR image as numpy array
        
    Returns:
        Processed frame with object bounding boxes
    """
    # Run inference
    results = model(frame, verbose=False, conf=0.4)
    
    # Get detections
    detections = results[0].boxes
    
    object_counts = {}
    
    for box in detections:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Get class and confidence
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        
        # Count objects
        object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        # Get color
        color = get_color(class_name)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw corner accents (industrial style)
        corner_len = 15
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)
        
        # Draw label
        label = f"{class_name} {confidence:.0%}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Label background
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Draw summary overlay
    total = sum(object_counts.values())
    text = f"OBJECTS: {total} detected"
    cv2.rectangle(frame, (10, 10), (220, 40), (0, 0, 0), -1)
    cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
    
    # Draw object counts sidebar
    y_offset = 50
    for obj_name, count in sorted(object_counts.items(), key=lambda x: -x[1])[:5]:
        color = get_color(obj_name)
        text = f"{count}x {obj_name}"
        cv2.rectangle(frame, (10, y_offset), (150, y_offset + 22), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, y_offset + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 26
    
    return frame
