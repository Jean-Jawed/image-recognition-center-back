"""
Face Mesh Processor
Detects facial landmarks (468 points) using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Create face mesh detector
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Custom drawing specs
TESSELATION_STYLE = mp_drawing.DrawingSpec(
    color=(80, 80, 80),
    thickness=1,
)

CONTOUR_STYLE = mp_drawing.DrawingSpec(
    color=(0, 180, 255),  # Amber/gold
    thickness=1,
)

IRIS_STYLE = mp_drawing.DrawingSpec(
    color=(0, 255, 255),  # Yellow
    thickness=1,
    circle_radius=1,
)


def process(frame: np.ndarray) -> np.ndarray:
    """
    Process a frame and draw face mesh landmarks.
    
    Args:
        frame: BGR image as numpy array
        
    Returns:
        Processed frame with face mesh drawn
    """
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = face_mesh.process(rgb_frame)
    
    # Draw face mesh if detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw tesselation (mesh triangles)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=TESSELATION_STYLE,
            )
            
            # Draw face contours
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=CONTOUR_STYLE,
            )
            
            # Draw irises
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=IRIS_STYLE,
            )
        
        # Add face count overlay
        face_count = len(results.multi_face_landmarks)
        text = f"FACES: {face_count} | 468 landmarks each"
        cv2.rectangle(frame, (10, 10), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
    
    return frame
