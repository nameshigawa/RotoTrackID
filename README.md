# RotoTrackID (Experimental)

## Demo

![RotoTrackID Demo](rototrackid_demo.gif)

> Select an ID → Export → Get clean alpha mattes

**RotoTrackID** is an experimental video rotoscoping tool that generates  
**alpha-matted PNG image sequences** from video using  
**YOLO object tracking + Segment Anything Model (SAM)**.

Instead of manual roto work, you simply select a **tracking ID** and export
a clean alpha matte for that object across the entire video.

This project is designed for VFX, compositing, and technical artists.

---

## ⚠ Experimental Status

RotoTrackID is under active development.

- Matte quality depends on the selected **SAM model**
- Analysis can be slow for long or high-resolution videos
# RotoTrackID (Experimental)

RotoTrackID is an experimental tool for producing per-object alpha mattes
from video. It combines YOLO-based tracking (with ByteTrack) and the
Segment Anything Model (SAM) to export RGBA PNG image sequences for a
selected tracking ID.

> Intended users: VFX artists, compositors, and technical artists.

## Status

This project is under active development. Results and performance depend on
the chosen SAM model, video resolution, and scene complexity. Some frames
may require manual cleanup.

## Features

- YOLO + ByteTrack object tracking
- Object selection via tracking ID
- High-quality segmentation using SAM
- Export of RGBA (alpha-enabled) PNG image sequences
- GUI-driven workflow (no model training required)

## Repository layout

Typical repository layout:

- `README.md` — this file
- `requirements.txt` — Python dependencies
- `main.py` — application entry point (GUI)
- `models/` — suggested location for downloaded model weight files (YOLO, SAM)
- `config/` — configuration files (e.g. `custom_bytetrack.yaml`)
- `examples/` — optional sample videos and exported frames

Adjust paths as needed; the app will look for model/checkpoint files in the
project root or configured model paths.

## Installation

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Download model weights and place them in the project root (not included
   in this repository):

- YOLO model example: `yolo11n.pt` (from Ultralytics)
- SAM model example: `sam_b.pt` (from the SAM distribution)

Refer to the official model pages for download links and licensing.

## Quick start

1. Launch the application:

```bash
python3 main.py
```

2. Select an input video in the GUI and run analysis. The app will:

- Track objects across frames and assign tracking IDs
- Optionally write an annotated preview video (boxes + labels)
- Export a JSON mapping of tracking IDs to labels and frame counts

Example JSON (`input_id_list.json`):

```json
{
  "4": { "label": "dog", "frames": 182 },
  "7": { "label": "person", "frames": 240 }
}
```

3. Select a tracking ID and choose the SAM model, then click **Export Alpha**.

- Output folder example: `rgba_id4/`
- Files: `frame_00001.png`, `frame_00002.png`, ...
- Each image is an 8-bit RGBA PNG with an alpha channel for the selected
  object and a transparent background.

These assets can be imported directly into compositing tools (After Effects,
Nuke, Blender, DaVinci Resolve, etc.).

## Notes & Tips

- Matte quality depends on the SAM model. Try different SAM checkpoints if
  available.
- Processing time increases with resolution and video length; consider
  testing on shorter clips or lower resolution during iteration.
- For problematic frames, refine the mask manually in a compositing tool.

### Configuration

The tracker uses a ByteTrack configuration file if present. This project
supports a custom configuration file named `custom_bytetrack.yaml` placed in
the `config/` directory (or project root). Use `custom_bytetrack.yaml` to
override ByteTrack parameters such as `track_thresh`, `match_thresh`, and
other tracker-specific options. Example location:

- `config/custom_bytetrack.yaml`

If you place a custom YAML file, the application will prefer it over default
tracker settings.
## License & acknowledgements

This repository does not include trained model weights.

- Uses Ultralytics YOLO (Apache-2.0)
- Uses Meta AI Segment Anything (check SAM model licenses)

Acknowledgements: Ultralytics, Meta AI (SAM)

---

If you'd like, I can also:

- Add usage screenshots to `README.md`
- Provide a small `examples/` folder with a sample video and exported
  frames
- Verify and pin the packages in `requirements.txt`