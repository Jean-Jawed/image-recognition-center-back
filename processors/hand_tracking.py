"""
Hand Tracking Processor
Detects hands and draws 21 landmarks per hand using MediaPipe Tasks API
"""

import cv2
import numpy as np
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path to the model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'hand_landmarker.task')

# Initialize detector (lazy loading)
_detector = None


def _get_detector():
    """Lazy initialization of the hand landmarker."""
    global _detector
    if _detector is None:
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _detector = vision.HandLandmarker.create_from_options(options)
    return _detector


# Colors for industrial look (BGR format for OpenCV)
LANDMARK_COLOR = (170, 255, 0)      # Cyan-green
CONNECTION_COLOR = (140, 200, 0)    # Darker cyan-green
TEXT_COLOR = (170, 255, 0)          # Cyan-green


def _draw_landmarks_on_image(frame: np.ndarray, detection_result) -> np.ndarray:
    """
    Draw hand landmarks on the frame.
    
    Args:
        frame: BGR image as numpy array
        detection_result: HandLandmarkerResult from MediaPipe Tasks
        
    Returns:
        Frame with landmarks drawn
    """
    if not detection_result.hand_landmarks:
        return frame
    
    h, w, _ = frame.shape
    
    # Process each detected hand
    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        # Get handedness (Left/Right)
        handedness = detection_result.handedness[idx]
        hand_label = handedness[0].category_name
        hand_score = handedness[0].score
        
        # Convert landmarks to pixel coordinates for custom drawing
        landmark_points = []
        for landmark in hand_landmarks:
            px = int(landmark.x * w)
            py = int(landmark.y * h)
            landmark_points.append((px, py))
        
        # Draw connections (MediaPipe hand connections)
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm connections
            (5, 9), (9, 13), (13, 17),
        ]
        
        for start_idx, end_idx in connections:
            start_point = landmark_points[start_idx]
            end_point = landmark_points[end_idx]
            cv2.line(frame, start_point, end_point, CONNECTION_COLOR, 2)
        
        # Draw landmark points
        for point in landmark_points:
            cv2.circle(frame, point, 4, LANDMARK_COLOR, -1)
            cv2.circle(frame, point, 5, (0, 0, 0), 1)  # Black outline
        
        # Draw hand label near wrist (landmark 0)
        wrist = landmark_points[0]
        label_text = f"{hand_label} {hand_score:.0%}"
        
        # Calculate text size for background
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw text background
        cv2.rectangle(
            frame,
            (wrist[0] - 5, wrist[1] - text_h - 10),
            (wrist[0] + text_w + 5, wrist[1] - 5),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            label_text,
            (wrist[0], wrist[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            TEXT_COLOR,
            2
        )
    
    return frame


def process(frame: np.ndarray) -> np.ndarray:
    """
    Process a frame and draw hand landmarks.
    
    Args:
        frame: BGR image as numpy array
        
    Returns:
        Processed frame with hand landmarks drawn
    """
    detector = _get_detector()
    
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect hand landmarks
    detection_result = detector.detect(mp_image)
    
    # Draw landmarks on frame
    frame = _draw_landmarks_on_image(frame, detection_result)
    
    return frame
