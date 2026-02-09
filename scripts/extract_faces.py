"""
Face frame extraction using SCRFD (InsightFace).
This script extracts aligned face frames from real and fake videos
to support frame-based deepfake detection.
Extracts up to 150 aligned face frames per video.
"""

import cv2
from pathlib import Path
from insightface.app import FaceAnalysis

# Prefer GPU if available, fallback to CPU
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

face_app = FaceAnalysis(name="buffalo_l", providers=providers)
face_app.prepare(ctx_id=0)

def extract_face_frames(video_path, output_dir, max_frames=150, reuse_window=5):
    """
    Extracts aligned face frames from a video using SCRFD.

    Args:
        video_path (Path): input video file
        output_dir (Path): directory to save extracted face frames
        max_frames (int): maximum number of frames to extract
        reuse_window (int): number of frames to reuse previous face bounding box
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️ Skipping unreadable video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"⚠️ No frames found in: {video_path}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Distance between sampled frames
    interval = max(1, total_frames // max_frames)

    count, saved = 0, 0
    prev_face = None
    reuse_counter = 0

    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:

            # Try reuse bounding box for a few frames (speed booster)
            if prev_face is not None and reuse_counter < reuse_window:
                bbox = prev_face['bbox'].astype(int)
                x1, y1, x2, y2 = bbox
                face = frame[y1:y2, x1:x2]
                if face.size != 0:
                    face = cv2.resize(face, (224, 224))
                    out_path = output_dir / f"{video_path.stem}_frame{saved:03}.jpg"
                    cv2.imwrite(str(out_path), face)
                    saved += 1
                    reuse_counter += 1
                    count += 1
                    continue

            # Run face detection
            faces = face_app.get(frame)

            if len(faces) > 0:
                face = faces[0]
                prev_face = face  # store for reuse
                reuse_counter = 0

                bbox = face['bbox'].astype(int)
                x1, y1, x2, y2 = bbox
                cropped = frame[y1:y2, x1:x2]

                if cropped.size != 0:
                    cropped = cv2.resize(cropped, (224, 224))
                    out_path = output_dir / f"{video_path.stem}_frame{saved:03}.jpg"
                    cv2.imwrite(str(out_path), cropped)
                    saved += 1
            else:
                # If detection fails, do nothing (skip)
                pass

        count += 1

    cap.release()

    print(f"✔ {saved}/{max_frames} face frames saved from {video_path.name}")


# ======================
# BATCH RUN
# ======================

input_root = Path("/content/deepfake-input-video")
output_root = Path("/content/deepfake-frames/150_frames")

fake_dir = input_root / "fake"
real_dir = input_root / "real"

fake_out_dir = output_root / "fake"
real_out_dir = output_root / "real"

for video_file in fake_dir.glob("*.mp4"):
    extract_face_frames(video_file, fake_out_dir, max_frames=150)

for video_file in real_dir.glob("*.mp4"):
    extract_face_frames(video_file, real_out_dir, max_frames=150)

