"""
Emotion Detection Processor
Detects facial emotions using FER (Facial Emotion Recognition)
"""

import cv2
import numpy as np
from fer import FER

# Initialize emotion detector
# Using MTCNN for better face detection
detector = FER(mtcnn=True)

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
    # Detect emotions
    results = detector.detect_emotions(frame)
    
    for result in results:
        # Get bounding box
        box = result["box"]
        x, y, w, h = box
        
        # Get emotions
        emotions = result["emotions"]
        
        # Find dominant emotion
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions[dominant_emotion]
        
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
            # Background bar
            cv2.rectangle(
                frame,
                (bar_x, bar_y + i * (bar_height + 4)),
                (bar_x + bar_width, bar_y + i * (bar_height + 4) + bar_height),
                (40, 40, 40),
                -1
            )
            
            # Score bar
            score_width = int(bar_width * score)
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
    text = f"EMOTION: {len(results)} face(s) detected"
    cv2.rectangle(frame, (10, 10), (280, 40), (0, 0, 0), -1)
    cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
    
    return frame
