"""GUI wrapper for alpha export tools.

This module provides a simple PyQt-based GUI that allows the user to:
- select a video file,
- run object tracking analysis to collect detected IDs and their labels,
- export per-ID alpha RGBA PNGs using SAM segmentation for selected IDs.

The GUI is intentionally minimal: analysis is performed by `analyze_video`, and
per-ID RGBA exports are handled by `export_alpha_by_id`.
"""

import os
import json
import time
from PySide6.QtWidgets import (
    QWidget, QPushButton, QLabel, QListWidget,
    QVBoxLayout, QFileDialog, QProgressBar, QCheckBox, QApplication,
    QLineEdit, QHBoxLayout
)
from analyze import analyze_video
from alpha_export import export_alpha_by_id


class AlphaToolGUI(QWidget):
    """Main GUI for alpha extraction.

    UI elements:
    - `btn_video`: open a file dialog to pick an input video
    - `btn_analyze`: run `analyze_video` to discover tracked IDs
    - `list_ids`: show discovered IDs; select one to export
    - `btn_export`: run `export_alpha_by_id` for selected ID
    - `progress` / `lbl_time`: show progress and timing information

    The GUI delegates analysis and export work to functions in the
    `alpha_tool` package and does not perform heavy computation itself.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RotoTrackID Alpha Export Tool")

        # Buttons for user actions
        self.btn_video = QPushButton("Select Video")
        # Checkbox to optionally write an annotated MP4 during analysis
        self.chk_annotated = QCheckBox("Write annotated video (boxes + labels)")
        # Line edit + browse button to choose SAM model path (default sam_b.pt)
        self.le_sam = QLineEdit("sam_b.pt")
        self.btn_browse_sam = QPushButton("Browse")
        self.btn_browse_sam.clicked.connect(self.browse_sam)
        self.btn_analyze = QPushButton("Analyze")
        self.btn_export = QPushButton("Export Alpha")

        # List of detected IDs and simple progress indicators
        self.list_ids = QListWidget()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.lbl_time = QLabel("")

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.btn_video)
        layout.addWidget(self.chk_annotated)
        # SAM model selector row
        row = QHBoxLayout()
        row.addWidget(QLabel("SAM model:"))
        row.addWidget(self.le_sam)
        row.addWidget(self.btn_browse_sam)
        layout.addLayout(row)
        layout.addWidget(self.btn_analyze)
        layout.addWidget(self.list_ids)
        layout.addWidget(self.btn_export)
        layout.addWidget(self.progress)
        layout.addWidget(self.lbl_time)

        # Connect signals to handlers
        self.btn_video.clicked.connect(self.select_video)
        self.btn_analyze.clicked.connect(self.run_analyze)
        self.btn_export.clicked.connect(self.run_export)

        # State
        self.video_path = None
        self.id_info = {}

    def select_video(self):
        """Open a file dialog to select the input video file.

        The selected path is stored in `self.video_path` for later operations.
        """
        self.video_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video (*.mp4 *.mov)"
        )

    def browse_sam(self):
        """Open a file dialog to choose a SAM model file and set it in the line edit."""
        path, _ = QFileDialog.getOpenFileName(self, "Select SAM model", "", "Model files (*.pt *.pth);;All files (*)")
        if path:
            self.le_sam.setText(path)

    def run_analyze(self):
        """Run `analyze_video` on the selected video and populate `list_ids`.

        The method provides a small `progress_cb` to update the progress bar and
        an ETA label while analysis runs. After analysis completes, results are
        saved to a JSON file next to the input video.
        """
        self.list_ids.clear()

        # Progress callback used by analyze_video to update UI
        def progress_cb(cur, total, elapsed, remain):
            # Guard against missing total metadata
            if total and total > 0:
                pct = int(cur / total * 100)
            else:
                pct = 0
            self.progress.setValue(pct)
            # Display current/total, elapsed and remaining seconds
            try:
                self.lbl_time.setText(
                    f"{cur}/{total} | elapsed {elapsed:.1f}s | remaining {remain:.1f}s"
                )
            except Exception:
                self.lbl_time.setText(f"{cur}/{total}")
            # Keep the UI responsive while analysis runs in the same thread
            QApplication.processEvents()

        # Prepare per-video output directory and ensure it exists. All
        # generated files (annotated MP4, JSON, RGBA folders) will live here.
        base_dir = os.path.dirname(self.video_path) or "."
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_dir = os.path.join(base_dir, f"{base_name}_outputs")
        os.makedirs(output_dir, exist_ok=True)

        # Delegate heavy work to analyze_video; it returns id->info mapping
        self.id_info = analyze_video(
            self.video_path,
            progress_cb,
            write_annotated=self.chk_annotated.isChecked()
        )

        # Populate the ID list with label information
        for tid, info in self.id_info.items():
            self.list_ids.addItem(f"ID {tid} : {info['label']}")

        # Save the ID list as JSON inside the per-video outputs folder
        json_path = os.path.join(output_dir, f"{base_name}_id_list.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.id_info, f, indent=2, ensure_ascii=False)

    def run_export(self):
        """Export RGBA PNGs for the currently selected ID in the list.

        The function reads the selected list item to obtain the tracker ID,
        creates an output directory `rgba_id<id>` next to the source video,
        and calls `export_alpha_by_id` with a simple progress callback.
        """
        item = self.list_ids.currentItem()
        if not item:
            return

        # Extract numeric ID from the list item text: "ID <tid> : <label>"
        tid = int(item.text().split()[1])
        base_dir = os.path.dirname(self.video_path) or "."
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_dir = os.path.join(base_dir, f"{base_name}_outputs")
        out_dir = os.path.join(output_dir, f"rgba_id{tid}")
        os.makedirs(out_dir, exist_ok=True)

        # Call export function; update the UI progress bar as export proceeds
        # Track start time so we can show elapsed and estimate remaining
        export_start = time.time()

        def export_progress(cur, total):
            if total and total > 0:
                pct = int(cur / total * 100)
            else:
                pct = 0
            self.progress.setValue(pct)
            # Compute elapsed and estimate remaining locally
            elapsed = time.time() - export_start
            fps = (cur / elapsed) if elapsed > 0 and cur > 0 else 0
            remain = (total - cur) / fps if fps > 0 else 0
            try:
                self.lbl_time.setText(f"{cur}/{total} | elapsed {elapsed:.1f}s | remaining {remain:.1f}s")
            except Exception:
                self.lbl_time.setText(f"{cur}/{total}")
            QApplication.processEvents()

        export_alpha_by_id(
            self.video_path,
            tid,
            out_dir,
            progress_cb=export_progress,
            sam_model=self.le_sam.text()
        )
