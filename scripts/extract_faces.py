"""
Face frame extraction using SCRFD (InsightFace).
Extracts up to 150 aligned face frames per video.
"""

import cv2
from pathlib import Path
from insightface.app import FaceAnalysis

