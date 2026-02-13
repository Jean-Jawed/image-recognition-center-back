"""
Hand Tracking Processor
Detects hands and draws 21 landmarks per hand using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Create hands detector (initialized once)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Custom drawing specs for industrial look
LANDMARK_STYLE = mp_drawing.DrawingSpec(
    color=(0, 255, 170),  # Cyan-green
    thickness=2,
    circle_radius=3,
)

CONNECTION_STYLE = mp_drawing.DrawingSpec(
    color=(0, 200, 140),
    thickness=2,
)


def process(frame: np.ndarray) -> np.ndarray:
    """
    Process a frame and draw hand landmarks.
    
    Args:
        frame: BGR image as numpy array
        
    Returns:
        Processed frame with hand landmarks drawn
    """
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = hands.process(rgb_frame)
    
    # Draw landmarks if hands detected
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            # Draw landmarks and connections
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                LANDMARK_STYLE,
                CONNECTION_STYLE,
            )
            
            # Add hand label (Left/Right)
            label = handedness.classification[0].label
            confidence = handedness.classification[0].score
            
            # Get wrist position for label placement
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            h, w, _ = frame.shape
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            
            # Draw label background
            text = f"{label} {confidence:.0%}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (cx - 5, cy - text_h - 10), (cx + text_w + 5, cy - 5), (0, 0, 0), -1)
            cv2.putText(frame, text, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 170), 2)
    
    return frame
