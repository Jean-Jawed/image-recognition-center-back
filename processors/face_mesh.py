"""
Face Mesh Processor
Detects facial landmarks (478 points including iris) using MediaPipe Tasks API
"""

import cv2
import numpy as np
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path to the model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'face_landmarker.task')

# Initialize detector (lazy loading)
_detector = None


def _get_detector():
    """Lazy initialization of the face landmarker."""
    global _detector
    if _detector is None:
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        _detector = vision.FaceLandmarker.create_from_options(options)
    return _detector


# Colors for industrial look (BGR format for OpenCV)
TESSELATION_COLOR = (80, 80, 80)    # Dark gray for mesh
CONTOUR_COLOR = (255, 180, 0)       # Amber/gold for contours
IRIS_COLOR = (255, 255, 0)          # Yellow for iris


# Face mesh contour indices (key facial features)
# These are the main contour landmarks for face outline, lips, eyes, eyebrows
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

LEFT_EYE = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
    386, 385, 384, 398
]

RIGHT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
    159, 160, 161, 246
]

LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

LIPS_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
    270, 269, 267, 0, 37, 39, 40, 185
]

LIPS_INNER = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
    310, 311, 312, 13, 82, 81, 80, 191
]

# Left iris landmarks (468-472)
LEFT_IRIS = [468, 469, 470, 471, 472]
# Right iris landmarks (473-477)
RIGHT_IRIS = [473, 474, 475, 476, 477]


def _draw_contour(frame: np.ndarray, landmarks: list, indices: list, 
                  color: tuple, thickness: int = 1, closed: bool = True) -> None:
    """Draw a contour connecting the specified landmark indices."""
    if not indices:
        return
    
    points = []
    for idx in indices:
        if idx < len(landmarks):
            points.append(landmarks[idx])
    
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, thickness)
        if closed and len(points) > 2:
            cv2.line(frame, points[-1], points[0], color, thickness)


def _draw_iris(frame: np.ndarray, landmarks: list, indices: list, color: tuple) -> None:
    """Draw iris circle."""
    if len(landmarks) <= max(indices, default=0):
        return
    
    # Center of iris is the first index
    center_idx = indices[0]
    if center_idx < len(landmarks):
        center = landmarks[center_idx]
        
        # Calculate radius from other iris points
        if len(indices) > 1:
            radii = []
            for idx in indices[1:]:
                if idx < len(landmarks):
                    pt = landmarks[idx]
                    r = int(np.sqrt((pt[0] - center[0])**2 + (pt[1] - center[1])**2))
                    radii.append(r)
            if radii:
                radius = int(np.mean(radii))
                cv2.circle(frame, center, radius, color, 1)
                cv2.circle(frame, center, 2, color, -1)  # Center dot


def _draw_tesselation(frame: np.ndarray, landmarks: list, color: tuple) -> None:
    """Draw face mesh tesselation (simplified - just some triangles)."""
    # Instead of full tesselation (which requires 468 triangle definitions),
    # we'll draw a simplified mesh by connecting nearby points
    n = len(landmarks)
    if n < 468:
        return
    
    # Draw a sparse mesh for visual effect without overwhelming
    # Connect every 5th point to nearby points
    step = 6
    for i in range(0, min(468, n), step):
        for j in range(i + step, min(468, n), step):
            # Only connect points that are close to each other
            p1 = landmarks[i]
            p2 = landmarks[j]
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if dist < 30:  # Only connect nearby points
                cv2.line(frame, p1, p2, color, 1)


def _draw_landmarks_on_image(frame: np.ndarray, detection_result) -> np.ndarray:
    """
    Draw face mesh landmarks on the frame.
    
    Args:
        frame: BGR image as numpy array
        detection_result: FaceLandmarkerResult from MediaPipe Tasks
        
    Returns:
        Frame with face mesh drawn
    """
    if not detection_result.face_landmarks:
        return frame
    
    h, w, _ = frame.shape
    
    # Process each detected face
    for face_landmarks in detection_result.face_landmarks:
        # Convert landmarks to pixel coordinates
        landmark_points = []
        for landmark in face_landmarks:
            px = int(landmark.x * w)
            py = int(landmark.y * h)
            landmark_points.append((px, py))
        
        # Draw tesselation (mesh triangles) - simplified
        _draw_tesselation(frame, landmark_points, TESSELATION_COLOR)
        
        # Draw face contours
        _draw_contour(frame, landmark_points, FACE_OVAL, CONTOUR_COLOR, 1, True)
        
        # Draw eyes
        _draw_contour(frame, landmark_points, LEFT_EYE, CONTOUR_COLOR, 1, True)
        _draw_contour(frame, landmark_points, RIGHT_EYE, CONTOUR_COLOR, 1, True)
        
        # Draw eyebrows
        _draw_contour(frame, landmark_points, LEFT_EYEBROW, CONTOUR_COLOR, 1, False)
        _draw_contour(frame, landmark_points, RIGHT_EYEBROW, CONTOUR_COLOR, 1, False)
        
        # Draw lips
        _draw_contour(frame, landmark_points, LIPS_OUTER, CONTOUR_COLOR, 1, True)
        _draw_contour(frame, landmark_points, LIPS_INNER, CONTOUR_COLOR, 1, True)
        
        # Draw irises (if available - landmarks 468-477)
        if len(landmark_points) >= 478:
            _draw_iris(frame, landmark_points, LEFT_IRIS, IRIS_COLOR)
            _draw_iris(frame, landmark_points, RIGHT_IRIS, IRIS_COLOR)
    
    # Draw info overlay
    face_count = len(detection_result.face_landmarks)
    info_text = f"FACES: {face_count} | 478 landmarks each"
    cv2.rectangle(frame, (10, 10), (320, 45), (0, 0, 0), -1)
    cv2.putText(
        frame,
        info_text,
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        CONTOUR_COLOR,
        2
    )
    
    return frame


def process(frame: np.ndarray) -> np.ndarray:
    """
    Process a frame and draw face mesh landmarks.
    
    Args:
        frame: BGR image as numpy array
        
    Returns:
        Processed frame with face mesh drawn
    """
    detector = _get_detector()
    
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect face landmarks
    detection_result = detector.detect(mp_image)
    
    # Draw landmarks on frame
    frame = _draw_landmarks_on_image(frame, detection_result)
    
    return frame
