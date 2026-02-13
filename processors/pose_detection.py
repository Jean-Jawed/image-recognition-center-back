"""
Pose Detection Processor
Detects human body pose with 33 landmarks using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Create pose detector
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Custom drawing specs for industrial look
LANDMARK_STYLE = mp_drawing.DrawingSpec(
    color=(255, 100, 0),  # Orange
    thickness=3,
    circle_radius=4,
)

CONNECTION_STYLE = mp_drawing.DrawingSpec(
    color=(200, 80, 0),
    thickness=2,
)


def process(frame: np.ndarray) -> np.ndarray:
    """
    Process a frame and draw pose landmarks.
    
    Args:
        frame: BGR image as numpy array
        
    Returns:
        Processed frame with pose skeleton drawn
    """
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = pose.process(rgb_frame)
    
    # Draw pose landmarks if detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            LANDMARK_STYLE,
            CONNECTION_STYLE,
        )
        
        # Add visibility indicator
        visible_count = sum(
            1 for lm in results.pose_landmarks.landmark 
            if lm.visibility > 0.5
        )
        total = len(results.pose_landmarks.landmark)
        
        # Draw info overlay
        text = f"POSE: {visible_count}/{total} points"
        cv2.rectangle(frame, (10, 10), (200, 40), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
    
    return frame
