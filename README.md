## DD2464 SNN Data Processing

This repository provides scripts and utilities to prepare training data for a spiking neural network (SNN) from event camera streams together with RGB video, including ROI cropping, event frame construction, homography calibration, and label transfer/visualization.

### 1. Environment setup

Recommended: create a clean Python virtual environment and install dependencies via `requirements.txt`.

- Python: 3.10+ recommended
- CUDA: requirements are pinned to CUDA 11.8 wheels in `requirements.txt`

Steps with Conda (works on Windows, Linux, and macOS):

```powershell
# 1) Create and activate a Conda environment
conda create -n snn-env python=3.10 -y
conda activate snn-env

# 2) Upgrade pip and install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you need a specific CUDA/PyTorch build, edit the first lines in `requirements.txt` accordingly. For CPU-only install, remove the `--extra-index-url` and use the CPU wheels from PyPI.

### 2. Data annotation workflow (overview)

The overall pipeline for preparing aligned event/video data and labels:

1) Prepare clips for both modalities
   - Pre-split your RGB video into fixed-length clips: `raw_data/.../processed/video_clips/clip_*.mp4`
   - Pre-split your event stream into fixed-length CSV clips: `raw_data/.../processed/event_clips/clip_*.csv`
   - Ensure a timestamps CSV exists for event frames aggregation: `raw_data/.../timestamps.csv`

2) Crop ROI on video clips
   - Edit and run `data_processing/crop_video_clip.py` to crop a spatial ROI from each video clip. This script also prints the corresponding ROI in event-frame coordinates to help keep both modalities consistent.
   - Key parameters inside the script:
     - `clip_id`: which `clip_*.mp4` to crop
     - `top_left` and `crop_size`: ROI on the video
   - Output: cropped clips under `raw_data/.../processed/cropped_video_clips/`.

3) Build event frame tensors
   - Edit and run `data_processing/create_event_frames.py` to aggregate events into dense frames over fixed time bins.
   - Key parameters inside the script:
     - `binning_interval` (ms), `clip_duration` (s), frame size `(frame_width, frame_height)`
     - `start_vedio_nr`, `end_video_nr`: which CSV clips to process
     - `dir2event_clips`, `path2timewindows`: where to read event CSVs and timestamps
   - Output: `raw_data/.../processed/event_frames/event_frames_{clip_id}.pt` and a `test_frame.png` preview.

4) Build Homography matrix (calibration)
   - Use `data_processing/kpt_selector.py` to select corresponding keypoints between several pairs of video frames and event frames.
   - Tips:
     - Prefer selecting multiple pairs from the same video across different time ranges; this usually yields a more robust mapping.
     - Ensure the ROI/crop used for the event frame matches the one used for video.
   - The script saves keypoint pairs into `H_matrix/homography_points_*.npz` and can help estimate a homography. These are later consumed by the labeling script.

5) Transfer labels from video to event frames (plus sanity-check visualization)
   - Run `data_processing/transfer_labels.py` to project detections (e.g., from YOLO on cropped video) onto event frames using the estimated homography and time alignment.
   - Internally this script:
     - Loads time alignment anchors from `time_alignments/<SESSION>/<clip_id>.txt`
     - Loads YOLO results from `raw_data/.../processed/yolo/result_cropped_{clip_id}/track/labels`
     - Loads event frames `raw_data/.../processed/event_frames/event_frames_{clip_id}.pt`
     - Loads homography keypoints from `H_matrix/`
     - Applies homography warp and generates training pairs; optionally visualizes for sanity check
   - Outputs:
     - Visualizations under `raw_data/.../processed/frame_with_labels/`
     - Cropped and labeled tensors under `raw_data/.../processed/frames_with_labels/{clip_id}/processed.pt`

### Notes and conventions

- Paths in scripts are currently set for a dataset under `raw_data/2024_09_10_16_00_41_recordings`. Adjust paths if your data lives elsewhere.
- Keep ROI settings consistent between video (`crop_video_clip.py`) and event (`transfer_labels.py` / `kpt_selector.py`). The video → event crop conversion helper is printed by `crop_video_clip.py`.
- YOLO inference utility exists in `data_processing/utility.py` (`run_yolo_on_clip` and `batch_process_yolo`) if you need to (re)generate detection results for your clips.
- Time alignment: anchors are parsed from `time_alignments/<SESSION>/<clip_id>.txt` via `parse_index_anchors`, and then expanded to per-frame mappings by `build_time_alignment_index_pairs`.

### Quick start checklist

1) Create Conda env and `pip install -r requirements.txt`
2) Verify your raw data directory structure matches the examples
3) Crop ROI with `data_processing/crop_video_clip.py`
4) Build event frames with `data_processing/create_event_frames.py`
5) Select keypoints with `data_processing/kpt_selector.py`
6) Transfer labels and visualize with `data_processing/transfer_labels.py`


### Plotting loss curves

Use the provided script to visualize training/validation losses from a JSON log:

```bash
python loss_curves.py path/to/log.json --smooth 3 --show
```

As an example with your naming preference:

```bash
python plot_losses.py log.json --smooth 3 --show
```

Notes:
- `json_path`: path to the JSON log containing `train_loss` and/or `validation_loss` arrays; optional fields like `Tau` are shown in the title if present.
- `--smooth`: moving average window size (epochs), e.g., `--smooth 3`.
- `--show`: open an interactive window after saving the figure.
- `--save path/to/fig.png` (optional): override the output path; by default it saves next to the JSON as `*.losses.png`.

