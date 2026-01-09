"""Video analysis helpers.

Provides `analyze_video` to run object tracking over a video and collect
per-tracker-ID statistics (label and count of frames seen).

The function is intentionally lightweight: it performs only tracking and
statistics collection and reports progress via a callback for UI integration.
"""

import cv2
import json
import os
import time
from collections import defaultdict
from ultralytics import YOLO


def analyze_video(video_path, progress_cb=None, write_annotated=False):
    """Analyze `video_path` with YOLO tracking and collect ID statistics.

    Parameters:
    - video_path (str): path to the input video file.
    - progress_cb (callable|None): optional callback called periodically with
      signature `progress_cb(current_frame, total_frames, elapsed_seconds, remaining_seconds)`.
    - write_annotated (bool): when True, writes an annotated video that draws
      all detected boxes and labels per frame to `<video_basename>_annot_analysis.mp4`.

    Returns:
    - dict mapping int id -> {"label": str, "frames": int}

    Notes:
    - Uses `ultralytics.YOLO.track` with `persist=True` so tracker state is
      carried between frames.
    - The function only reads the input video; when `write_annotated` is True
      it will also write an annotated MP4 next to the input video.
    """

    # Initialize the detection/tracking model
    yolo = YOLO("yolo11n.pt")

    # Open video capture and get total frame count for progress reporting
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare an output directory that starts with the input video's base name
    base_dir = os.path.dirname(video_path) or "."
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(base_dir, f"{base_name}_outputs")

    # Writer will be created lazily from the first decoded frame. Some files
    # lack reliable width/height metadata, so creating the writer after we
    # successfully read a frame is more robust.
    writer = None
    out_video = None

    # id_info holds discovered IDs -> {label, frames}
    id_info = defaultdict(lambda: {"label": "", "frames": 0})

    current = 0
    start = time.time()

    # Process frames sequentially
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current += 1

        # Run tracker on the frame. `persist=True` keeps tracker state.
        results = yolo.track(
            frame,
            persist=True,
            conf=0.3,
            tracker="custom_bytetrack.yaml",
            verbose=False
        )

        # If the caller requested an annotated video, create the writer the
        # first time we have a valid frame. This handles inputs missing
        # width/height metadata.
        if write_annotated and writer is None:
            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_video = os.path.join(output_dir, f"{base_name}_annot_analysis.mp4")
            try:
                os.makedirs(output_dir, exist_ok=True)
                writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))
            except Exception:
                writer = None

        # If requested, draw all detected boxes+labels on the frame for the
        # annotated output before collecting ID stats.
        if writer is not None and results:
            r0 = results[0]
            if r0.boxes is not None:
                try:
                    boxes = r0.boxes.xyxy.cpu().numpy()
                    ids = r0.boxes.id.cpu().numpy()
                    clss = r0.boxes.cls.cpu().numpy()
                except Exception:
                    boxes = r0.boxes.xyxy
                    ids = r0.boxes.id
                    clss = r0.boxes.cls

                for box, tid, cls in zip(boxes, ids, clss):
                    tid = int(tid)
                    label = yolo.names[int(cls)]
                    x1, y1, x2, y2 = map(int, box)
                    color = (int(tid * 37) % 256, int(tid * 17) % 256, int(tid * 97) % 256)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{label} ID:{tid}"
                    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
                    cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Collect ID stats from results
        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue

            for tid, cls in zip(
                r.boxes.id.cpu().numpy(),
                r.boxes.cls.cpu().numpy()
            ):
                tid = int(tid)
                id_info[tid]["label"] = yolo.names[int(cls)]
                id_info[tid]["frames"] += 1

        # Report progress to the optional callback (frame count, elapsed, remain)
        if progress_cb:
            elapsed = time.time() - start
            fps = current / elapsed if elapsed > 0 else 0
            remain = (total - current) / fps if fps > 0 else 0
            try:
                progress_cb(current, total, elapsed, remain)
            except Exception:
                # Keep analysis robust if callback fails
                pass

        # Write annotated frame (if any) after drawing
        if writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()
    return dict(id_info)

def run_tracking_for_colab(
    video_path,
    out_dir="result",
    yolo_model_path="yolo11n.pt",
    bytetrack_cfg="custom_bytetrack.yaml"
):
    """
    Colab helper:
    - Generate a preview video (preview.mp4) with ID-annotated bounding boxes
    - Save the list of detected IDs to a JSON file
    """

    os.makedirs(out_dir, exist_ok=True)

    preview_path = os.path.join(out_dir, "preview.mp4")
    tracks_path = os.path.join(out_dir, "tracks.json")

    # 動画サイズ取得
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(preview_path, fourcc, fps, (w, h))

    # 実際の解析を実行
    analyze_video(
        video_path=video_path,
        yolo_model_path=yolo_model_path,
        bytetrack_cfg=bytetrack_cfg,
        writer=writer,
        progress_cb=None
    )

    return preview_path, tracks_path
