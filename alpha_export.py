"""Alpha export helper.

This module provides a single function `export_alpha_by_id` that scans a video
and extracts per-frame RGBA PNGs (with alpha masks) for a given tracked ID.

Key behavior:
- Uses `ultralytics.YOLO.track` for object tracking and `ultralytics.SAM` for
  segmentation within the tracked bounding box.
- Writes PNG frames with RGBA channels into `out_dir` named `frame_XXXXX.png`.
- Optionally reports progress through `progress_cb(frame_index, total_frames)`.

Parameters for `export_alpha_by_id`:
 - `video_path` (str): path to the input video file.
 - `target_id` (int): the tracker ID to extract masks for.
 - `out_dir` (str): directory where PNG frames will be written.
 - `pad` (int): number of pixels to pad around the tracked bounding box before
    sending the crop to SAM (helps include context for segmentation).
 - `progress_cb` (callable): optional callback to receive progress updates.

The function performs no return value; it writes files to disk.
"""

import cv2
import json
import os
import numpy as np
from ultralytics import YOLO, SAM


def export_alpha_by_id(
    video_path,
    target_id,
    out_dir,
    pad=40,
    progress_cb=None,
    sam_model="sam_b.pt"
):
    """Extract RGBA frames for a single tracked ID from `video_path`.

    The function loops through the video, runs YOLO tracking and, when the
    `target_id` is present in a frame, crops a padded bbox and passes it to
    SAM for mask prediction. The mask is converted into an alpha channel and
    combined with the original RGB frame to produce an RGBA PNG.
    """

    # Initialize models (assumed to exist in working directory)
    yolo = YOLO("yolo11n.pt")
    sam = SAM(sam_model)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(out_dir, exist_ok=True)

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Increment frame counter early so output filenames match original code
        frame_idx += 1
        mask_frame = None

        # Run tracker on the current frame
        results = yolo.track(
            frame,
            persist=True,
            conf=0.3,
            tracker="custom_bytetrack.yaml",
            verbose=False
        )

        # If tracker found boxes, iterate and look for the target ID
        if results[0].boxes.id is not None:
            for box, tid in zip(
                results[0].boxes.xyxy,
                results[0].boxes.id
            ):
                if int(tid) != target_id:
                    continue

                # Convert box to integers and pad the region
                x1, y1, x2, y2 = map(int, box)
                h, w = frame.shape[:2]

                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)

                # Run SAM on the padded bbox to get masks
                sam_r = sam(frame, bboxes=[x1, y1, x2, y2], verbose=False)
                if sam_r[0].masks is not None:
                    mask_frame = sam_r[0].masks.data[0].cpu().numpy()

        # If we got a valid mask, compose RGBA and write PNG
        if mask_frame is not None:
            alpha = (mask_frame * 255).astype(np.uint8)
            rgba = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
            rgba[..., :3] = frame
            rgba[..., 3] = alpha
            rgba[..., :3] = (
                rgba[..., :3].astype(float) * (alpha[..., None] / 255)
            ).astype(np.uint8)

            out_path = os.path.join(out_dir, f"frame_{frame_idx:05d}.png")
            cv2.imwrite(out_path, rgba)

        # Optionally call progress callback with (current_frame, total_frames)
        if progress_cb:
            try:
                progress_cb(frame_idx, total)
            except Exception:
                # Ignore errors in user-provided callback
                pass

    cap.release()
