"""
Emotion Detection Processor
Detects facial emotions using DeepFace
"""

import cv2
import numpy as np
from deepface import DeepFace

# Emotion colors (industrial palette)
EMOTION_COLORS = {
    "angry": (0, 0, 200),      # Red
    "disgust": (0, 100, 0),    # Dark green
    "fear": (128, 0, 128),     # Purple
    "happy": (0, 200, 100),    # Green
    "sad": (200, 100, 0),      # Blue
    "surprise": (0, 200, 200), # Yellow
    "neutral": (128, 128, 128) # Gray
}


def process(frame: np.ndarray) -> np.ndarray:
    """
    Process a frame and detect emotions.
    
    Args:
        frame: BGR image as numpy array
        
    Returns:
        Processed frame with emotion annotations
    """
    try:
        # Detect emotions using DeepFace
        # enforce_detection=False prevents errors when no face is found
        # detector_backend="opencv" is fastest
        results = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',
            silent=True
        )
        
        # Ensure results is a list
        if not isinstance(results, list):
            results = [results]
        
    except Exception:
        # No face detected or error
        results = []
    
    for result in results:
        # Get bounding box (region)
        region = result.get("region", {})
        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 0)
        h = region.get("h", 0)
        
        # Skip if no valid region
        if w == 0 or h == 0:
            continue
        
        # Get emotions dict and dominant emotion
        emotions = result.get("emotion", {})
        dominant_emotion = result.get("dominant_emotion", "neutral")
        confidence = emotions.get(dominant_emotion, 0) / 100  # DeepFace returns 0-100
        
        # Get color for dominant emotion
        color = EMOTION_COLORS.get(dominant_emotion, (128, 128, 128))
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw emotion label with confidence
        label = f"{dominant_emotion.upper()}: {confidence:.0%}"
        
        # Label background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - text_h - 15), (x + text_w + 10, y - 5), color, -1)
        cv2.putText(frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw emotion bar chart on the side
        bar_x = x + w + 10
        bar_y = y
        bar_width = 80
        bar_height = 12
        
        for i, (emotion, score) in enumerate(sorted(emotions.items())):
            # Normalize score (DeepFace returns 0-100)
            score_normalized = score / 100
            
            # Background bar
            cv2.rectangle(
                frame,
                (bar_x, bar_y + i * (bar_height + 4)),
                (bar_x + bar_width, bar_y + i * (bar_height + 4) + bar_height),
                (40, 40, 40),
                -1
            )
            
            # Score bar
            score_width = int(bar_width * score_normalized)
            bar_color = EMOTION_COLORS.get(emotion, (128, 128, 128))
            cv2.rectangle(
                frame,
                (bar_x, bar_y + i * (bar_height + 4)),
                (bar_x + score_width, bar_y + i * (bar_height + 4) + bar_height),
                bar_color,
                -1
            )
            
            # Emotion label
            cv2.putText(
                frame,
                emotion[:3].upper(),
                (bar_x + bar_width + 5, bar_y + i * (bar_height + 4) + bar_height - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (180, 180, 180),
                1
            )
    
    # Add detection count overlay
    face_count = len([r for r in results if r.get("region", {}).get("w", 0) > 0])
    text = f"EMOTION: {face_count} face(s) detected"
    cv2.rectangle(frame, (10, 10), (280, 40), (0, 0, 0), -1)
    cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
    
    return frame