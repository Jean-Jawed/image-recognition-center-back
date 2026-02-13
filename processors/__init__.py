"""
IRC Processors Module
Each processor implements a process(frame) -> frame function
"""

from . import hand_tracking
from . import pose_detection
from . import face_mesh
from . import emotion_detection
from . import object_detection

__all__ = [
    "hand_tracking",
    "pose_detection", 
    "face_mesh",
    "emotion_detection",
    "object_detection",
]
