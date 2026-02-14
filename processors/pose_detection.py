"""
Pose Detection Processor
Detects human body pose with 33 landmarks using MediaPipe Tasks API
"""

import cv2
import numpy as np
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path to the model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'pose_landmarker_heavy.task')

# Initialize detector (lazy loading)
_detector = None


def _get_detector():
    """Lazy initialization of the pose landmarker."""
    global _detector
    if _detector is None:
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        _detector = vision.PoseLandmarker.create_from_options(options)
    return _detector


# Colors for industrial look (BGR format for OpenCV)
LANDMARK_COLOR = (0, 100, 255)      # Orange
CONNECTION_COLOR = (0, 80, 200)     # Darker orange
TEXT_COLOR = (0, 100, 255)          # Orange


# Pose connections (33 landmarks)
# Based on MediaPipe Pose topology
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),  # Right eye
    (0, 4), (4, 5), (5, 6), (6, 8),  # Left eye
    (9, 10),                          # Mouth
    # Torso
    (11, 12),  # Shoulders
    (11, 23), (12, 24), (23, 24),  # Shoulders to hips
    # Right arm
    (11, 13), (13, 15),  # Shoulder to wrist
    (15, 17), (15, 19), (15, 21), (17, 19),  # Hand
    # Left arm
    (12, 14), (14, 16),  # Shoulder to wrist
    (16, 18), (16, 20), (16, 22), (18, 20),  # Hand
    # Right leg
    (23, 25), (25, 27),  # Hip to ankle
    (27, 29), (27, 31), (29, 31),  # Foot
    # Left leg
    (24, 26), (26, 28),  # Hip to ankle
    (28, 30), (28, 32), (30, 32),  # Foot
]


def _draw_landmarks_on_image(frame: np.ndarray, detection_result) -> np.ndarray:
    """
    Draw pose landmarks on the frame.
    
    Args:
        frame: BGR image as numpy array
        detection_result: PoseLandmarkerResult from MediaPipe Tasks
        
    Returns:
        Frame with landmarks drawn
    """
    if not detection_result.pose_landmarks:
        return frame
    
    h, w, _ = frame.shape
    
    # Process each detected pose
    for pose_landmarks in detection_result.pose_landmarks:
        # Convert landmarks to pixel coordinates
        landmark_points = []
        visibility_scores = []
        
        for landmark in pose_landmarks:
            px = int(landmark.x * w)
            py = int(landmark.y * h)
            landmark_points.append((px, py))
            # Visibility indicates how likely the landmark is visible (not occluded)
            visibility_scores.append(landmark.visibility if hasattr(landmark, 'visibility') else 1.0)
        
        # Draw connections
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                # Only draw if both landmarks are reasonably visible
                if visibility_scores[start_idx] > 0.5 and visibility_scores[end_idx] > 0.5:
                    start_point = landmark_points[start_idx]
                    end_point = landmark_points[end_idx]
                    cv2.line(frame, start_point, end_point, CONNECTION_COLOR, 2)
        
        # Draw landmark points
        for i, point in enumerate(landmark_points):
            if visibility_scores[i] > 0.5:
                cv2.circle(frame, point, 5, LANDMARK_COLOR, -1)
                cv2.circle(frame, point, 6, (0, 0, 0), 1)  # Black outline
        
        # Count visible landmarks
        visible_count = sum(1 for v in visibility_scores if v > 0.5)
        total = len(landmark_points)
        
        # Draw info overlay
        info_text = f"POSE: {visible_count}/{total} points"
        cv2.rectangle(frame, (10, 10), (220, 45), (0, 0, 0), -1)
        cv2.putText(
            frame,
            info_text,
            (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            TEXT_COLOR,
            2
        )
    
    return frame


def process(frame: np.ndarray) -> np.ndarray:
    """
    Process a frame and draw pose landmarks.
    
    Args:
        frame: BGR image as numpy array
        
    Returns:
        Processed frame with pose skeleton drawn
    """
    detector = _get_detector()
    
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect pose landmarks
    detection_result = detector.detect(mp_image)
    
    # Draw landmarks on frame
    frame = _draw_landmarks_on_image(frame, detection_result)
    
    return frame
