"""
IRC Processors Module (Lite Version)
Each processor implements a process(frame) -> frame function
"""

from . import hand_tracking
from . import pose_detection
from . import face_mesh

__all__ = [
    "hand_tracking",
    "pose_detection", 
    "face_mesh",
]
