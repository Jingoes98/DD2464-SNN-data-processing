import pandas as pd
import numpy as np
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import re
from matplotlib.widgets import Button

class PointSelector:
    def __init__(self, image1, image2, max_points=20, id=None, save_dir = "H_matrix"):
        self.image1 = image1
        self.image2 = image2
        self.max_points = max_points
        self.points1 = []
        self.points2 = []
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 7))
        self.ax1.imshow(self.image1, cmap='gray')
        self.ax2.imshow(self.image2, cmap='gray')
        self.ax1.set_title("① Frame-based pic")
        self.ax2.set_title("② Event Frame pic")
        self.cid1 = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.done = False
        self.id = id
        self.save_dir = save_dir

        ax_done = plt.axes([0.45, 0.01, 0.1, 0.05])
        self.btn_done = Button(ax_done, 'Done')
        self.btn_done.on_clicked(self.finish)

    def onclick(self, event):
        if event.inaxes == self.ax1 and len(self.points1) < self.max_points:
            self.points1.append((event.xdata, event.ydata))
            self.ax1.plot(event.xdata, event.ydata, 'ro')
            self.fig.canvas.draw()
        elif event.inaxes == self.ax2 and len(self.points2) < self.max_points:
            self.points2.append((event.xdata, event.ydata))
            self.ax2.plot(event.xdata, event.ydata, 'go')
            self.fig.canvas.draw()

    def finish(self, event):
        if len(self.points1) == len(self.points2) and len(self.points1) >= 4:
            file_name = f"homography_points_{self.id}.npz" if self.id is not None else "homography_points.npz"
            save_path = os.path.join(self.save_dir, file_name)
            np.savez(save_path, src=np.array(self.points1), dst=np.array(self.points2))
            print(f"saved as {file_name}")
            plt.close(self.fig)
        else:
            print("unequal number of points!")

def aggregate_events_spatially(x_coords, y_coords, polarities, timestamps, time_windows, width, height, decay = False, decay_rate = 1):
    frames = []
    event_index = 0
    n_events = len(timestamps)

    for i in tqdm(range(len(time_windows) - 1)):
        start_time = time_windows[i - 1] # need to be looked into
        end_time = time_windows[i + 1]
        frame = np.zeros((height, width))

        while event_index < n_events and timestamps[event_index] < end_time:
            if timestamps[event_index] >= start_time:
                event_x = x_coords[event_index]
                event_y = y_coords[event_index]
                polarity = polarities[event_index]
                time_since_start = timestamps[event_index] - start_time
                if 0 <= event_x < width and 0 <= event_y < height:
                    if decay:
                        decay_multiplier = np.exp(-time_since_start/decay_rate)
                        frame[event_y, event_x] += polarity * decay_multiplier
                    else:
                        frame[event_y, event_x] = polarity
            event_index += 1

        frames.append(torch.tensor(frame, dtype=torch.float32).to_sparse())
    
    return torch.stack(frames) # no need for coalescing as we are not using this tensor for training

def event_frames_to_imgs(event_frames: torch.Tensor, output_dir: str, n_frames: int = 100):
    # generate images from event frames
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm(range(min(n_frames, event_frames.shape[0])), desc="Saving frames"):
        frame_dense = event_frames[i].to_dense()
        plt.imsave(os.path.join(output_dir, f'frame_{i}.png'), frame_dense, cmap='gray')

def run_yolo_on_clip(clip_path: str, clip_index: int, output_dir: str, save_root = "yolo", cropped = True):
    # Placeholder for YOLO inference code
    # This function should run YOLO on the given clip and save the results to output_dir
    model = YOLO('yolov8n.pt')  # Load a pre-trained YOLO model
    target_classes = {2, 5, 7} # Define target classes for detection 2, 5, 7
    cap = cv2.VideoCapture(clip_path)
    frame_index = 0
    clip_index = str(clip_index)
    save_dir = Path(output_dir) / save_root
    if cropped:
        save_label_dir = save_dir / f"result_cropped_{clip_index}" / "track" / "labels"
        save_vis_dir = save_dir / f"result_cropped_{clip_index}" / "vis"
    else:
        save_label_dir = save_dir / f"result_{clip_index}" / "track" / "labels"
        save_vis_dir = save_dir / f"result_{clip_index}" / "vis"
    save_label_dir.mkdir(parents=True, exist_ok=True)
    save_vis_dir.mkdir(parents=True, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)[0]
        lines = []
        vis_frame = frame.copy()    
        for box in results.boxes:
            cls_id = int(box.cls.item())
            if cls_id in target_classes:
                x_center, y_center, width, height = box.xywhn[0].tolist()
                line = f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                lines.append(line)
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(vis_frame, str(cls_id), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        filename_label = save_label_dir / f"img{frame_index:04d}.txt"
        with open(filename_label, 'w') as f:
            f.write('\n'.join(lines))

        vis_path = save_vis_dir / f"img{frame_index:04d}.png"
        cv2.imwrite(str(vis_path), vis_frame)

        frame_index += 1

def batch_process_yolo(clips_dir: str):
    # Batch process YOLO on all clips in the specified directory
    clips = sorted(Path(clips_dir).glob("clip_*.mp4"), key=lambda x: int(x.stem.split('_')[1]))
    for clip in clips:
        clip_index = int(clip.stem.split('_')[1])  # clip_1 -> 1
        print(f"Processing clip_{clip_index}.mp4...")
        run_yolo_on_clip(clip, clip_index, output_dir="raw_data/2024_09_10_16_00_41_recordings/processed")

def _parse_frame_index(fn: str):
    """parse the frame index from the filename."""
    m = re.search(r'(\d+)\.txt$', fn)
    if m:
        return int(m.group(1))
    # backward compatibility for old filenames
    try:
        return int(fn[3:-4])
    except Exception:
        raise ValueError(f"Cannot parse frame index from filename: {fn}")

def parse_index_anchors(path2txt, required_min_pairs: int = 2):
    pairs = []

    with open(path2txt, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            nums = re.findall(r'-?\d+', line)
            if len(nums) < 2:
                continue
        
            e = int(nums[0])
            v = int(nums[1])
            pairs.append((e, v))

    if not pairs:
        raise ValueError(f"No valid anchor pairs found in: {path2txt}")
    
    pairs = sorted(dict.fromkeys(pairs), key=lambda x: x[0])

    if len(pairs) < required_min_pairs:
        raise ValueError(
            f"Need at least {required_min_pairs} anchor pairs, got {len(pairs)} from {path2txt}"
        )
    e_anchors = np.array([p[0] for p in pairs], dtype=int)
    v_anchors = np.array([p[1] for p in pairs], dtype=int)
    return e_anchors, v_anchors

# def build_pairs_from_event_windows(
#         event_start: int, # starting index of event frames
#         event_count: int, # number of event frames to be matched
#         *,
#         event_fps: int = 100,
#         video_fps: int = 90,
#         video_start: int = 0, # the setting for known offset: event_start - video_start
#         max_event_frames: int = None, # valid indices [0, max_event_index]
#         max_video_frames: int = None, # valid indices [0, max_video_index]
#         prefer = "nearest"
# ):
#     """Build pairs of event and video indices based on the given parameters.
#     Args:
#         event_start: The starting index for events.
#         event_count: The number of events to process.
#         event_fps: Frames per second for events.
#         video_fps: Frames per second for video.
#         video_start: The starting index for video frames.
#         max_event_index: Maximum valid index for events.
#         max_video_index: Maximum valid index for video frames.
#     Returns:
#         A tuple of two lists: (event_indices, video_indices).
#     """
#     e_idx = event_start + np.arange(event_count, dtype=int)
#     event_end = event_start + event_count
#     if max_event_frames is not None:
#         e_idx = e_idx[(e_idx >= 0) & (e_idx < max_event_frames)]
#     e_duration = event_count / event_fps # time period of the number of event frames
#     video_count = int(e_duration * video_fps) # number of video frames of the same time period as for event frames
#     video_end = video_start + video_count
#     v_idx = np.round((e_idx - event_start) * (video_end - video_start) / (event_end - event_start)).astype(int) + video_start
#     # v_idx = video_start + np.arange(event_count, dtype=int)
    
#     if max_video_frames is not None:
#         v_idx = np.clip(v_idx, 0, max_video_frames - 1)
    
#     return e_idx, v_idx

def build_time_alignment_index_pairs(e_anchors, v_anchors, 
                                    rounding: str = "round" # "round"/"ceil"/"floor"
                                    ):
    """
    build Multi-segment linear mapping for given index anchors from event frames and video frames
    args:
        e_anchors: list of index of event frames
        v_anchors: list of index of video frames
    return:
        e_idx: index from e_anchors[0] to e_anchor[-1] in natural order
        v_idx: mapped index of video index
    """
    e = np.asarray(e_anchors, dtype=int)
    v = np.asarray(v_anchors, dtype=int)
    if e.shape != v.shape or e.ndim != 1 or e.size < 2:
        raise ValueError("anchors must be 1D, same length, and length >= 2") 
    order = np.argsort(e)
    e, v = e[order], v[order]
    if np.any(np.diff(e) <= 0):
        raise ValueError("event anchors must be strictly increasing")
    
    e_min, e_max = int(e[0]), int(e[-1])
    e_idx = np.arange(e_min, e_max + 1, dtype=int)
    v_idx = np.empty_like(e_idx)

    if rounding == "floor":
        R = np.floor
    elif rounding == "ceil":
        R = np.ceil
    else:
        R = np.rint

    write_mask_total = np.zeros_like(e_idx, dtype=bool)
    for i in range(len(e) - 1):
        ea, eb = int(e[i]), int(e[i+1])
        va, vb = int(v[i]), int(v[i+1])
        if eb <= ea:
            raise ValueError(f"bad segment: {ea} -> {eb}")

        if i < len(e) - 2:
            mask = (e_idx >= ea) & (e_idx < eb)
        else:
            mask = (e_idx >= ea) & (e_idx <= eb)
        if not np.any(mask):
            continue

        e_seg = e_idx[mask].astype(float)
        t = (e_seg - ea) / (eb - ea)
        v_cont = va + t * (vb - va)
        v_idx[mask] = R(v_cont).astype(int)
        write_mask_total[mask] = True

    if not write_mask_total.all():
        v_idx[~write_mask_total] = v_idx[write_mask_total][np.searchsorted(
            np.flatnonzero(write_mask_total),
            np.flatnonzero(~write_mask_total),
            side="right") - 1]

    v_idx[0]  = int(v[0])
    v_idx[-1] = int(v[-1])

    return e_idx, v_idx

def get_labels_with_indices(yolo_results_dir, 
                            v_idx,
                            target_classes: set = {2, 5, 7}):
    labels_by_video_idx = {}
    event_count = len(v_idx)
    for fn in os.listdir(yolo_results_dir):
        if not fn.endswith('.txt'):
            continue
        vid_idx = _parse_frame_index(fn)
        path = os.path.join(yolo_results_dir, fn)
        boxes = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(parts[0])
                except ValueError:
                    continue
                if cls in set(target_classes):
                    vals = [float(v) for v in parts[1:5]] # x, y, w, h in original video frame coordinates, normalized to [0, 1]
                    boxes.append(vals)
        labels_by_video_idx[vid_idx] = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)

    targets = np.empty(event_count, dtype=object)
    empty = torch.empty((0, 4), dtype=torch.float32)
    for i, v in enumerate(v_idx):
        targets[i] = labels_by_video_idx.get(int(v), empty)
    return targets
    

def get_labels_aligned_to_events(
        yolo_results_dir: str,
        event_start: int, event_count: int,
        *,
        event_fps: int = 100,
        video_fps: int = 90,
        video_start: int = 0,
        max_event_frames: int = 5999,  # default to 5999 for 60 seconds at 100 fps
        max_video_frames: int = 5399,  # default to 5399 for 60 seconds at 90 fps
        target_classes: set = {2, 5, 7},
        return_mapping: bool = True,
):
    '''Get YOLO labels aligned to event frames.'''
    event_indices, video_indices = build_pairs_from_event_windows(
        event_start, event_count,
        event_fps=event_fps, video_fps=video_fps,
        video_start=video_start,
        max_event_frames=max_event_frames,
        max_video_frames=max_video_frames,
    )

    labels_by_video_idx = {}
    for fn in os.listdir(yolo_results_dir):
        if not fn.endswith('.txt'):
            continue
        vid_idx = _parse_frame_index(fn)
        path = os.path.join(yolo_results_dir, fn)
        boxes = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(parts[0])
                except ValueError:
                    continue
                if cls in set(target_classes):
                    vals = [float(v) for v in parts[1:5]] # x, y, w, h in original video frame coordinates, normalized to [0, 1]
                    boxes.append(vals)
        labels_by_video_idx[vid_idx] = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)

    targets = np.empty(event_count, dtype=object)
    empty = torch.empty((0, 4), dtype=torch.float32)
    for i, v in enumerate(video_indices):
        targets[i] = labels_by_video_idx.get(int(v), empty)
    if return_mapping:
        mapping = {
            "event_indices": event_indices,  # aligned event indices
            "video_indices": video_indices,  # aligned video indices
        }
        return targets, mapping
    return targets
# def get_labels_from_yolo_results(
#         yolo_results_dir: str,
#         target_length: int,
#         video_fps: int = 90,
#         event_fps: int = 100,
#         video_start_frame: int = 0,
#         event_start_frame: int = 0,
#         n_video_frames: int = None, 
#         target_classes: set = {2, 5, 7},
#         return_mapping: bool = False,
# ):
#     labels_by_video_idx = {}
#     for fn in os.listdir(yolo_results_dir):
#         if not fn.endswith('.txt'):
#             continue
#         vid_idx = _parse_frame_index(fn)
#         path = os.path.join(yolo_results_dir, fn)
#         boxes = []
#         with open(path, 'r') as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) < 5:
#                     continue
#                 try:
#                     cls = int(parts[0])
#                 except ValueError:
#                     continue
#                 if cls in set(target_classes):
#                     vals = [float(v) for v in parts[1:5]] # x, y, w, h in original video frame coordinates
#                     boxes.append(vals)
#         if boxes:
#             labels_by_video_idx[vid_idx] = torch.tensor(boxes, dtype=torch.float32)
#         else:
#             labels_by_video_idx[vid_idx] = torch.empty((0, 4), dtype=torch.float32)
        
#     event_indices = event_start_frame + np.arange(target_length, dtype=int)
#     scale = float(video_fps) / float(event_fps)
#     video_indices = video_start_frame + np.rint((event_indices - event_start_frame) * scale).astype(int)
#     if n_video_frames is not None and n_video_frames > 0:
#         video_indices = np.clip(video_indices, 0, n_video_frames - 1)

#     target_tensors = np.empty(target_length, dtype=object)
#     empty = torch.empty((0, 4), dtype=torch.float32)
#     for i, v_idx in enumerate(video_indices):
#         target_tensors[i] = labels_by_video_idx.get(int(v_idx), empty)
        
#     if return_mapping:
#         mapping = {
#             "event_indices": event_indices,  
#             "video_indices": video_indices,  
#         }
#         return target_tensors, mapping
#     return target_tensors
                
def extract_event_subset(event_frames, event_indices_np):
    N = event_frames.shape[0]
    idx = torch.as_tensor(event_indices_np, dtype=torch.long)
    valid = (idx >= 0) & (idx < N)
    if valid.sum().item() != idx.numel():
        dropped = (~valid).sum().item()
        print(f"[warn] {dropped} indices out of range [0, {N-1}] -> will be dropped")
    idx = idx[valid]
    subset = event_frames.index_select(0, idx)
    return subset, idx

def warp_points_xy(points_src, H_norm, src_wh, dst_wh, Normalize=False):
    '''
    Warp points from source to destination using a normalized homography matrix.
    Args:
        points_norm: (N, 2) or (2, ) array of normalized/pixel source points (x, y)
        H_norm: (3, 3) normalized homography matrix
        src_wh: (width, height) of the source frame
        dst_wh: (width, height) of the destination frame
        Normalize: whether to normalize the source points by src_wh, if True, the points_src should be normalized
    Returns:
        proj: (N, 2) array of warped destination points (x, y)
    '''
    if isinstance(points_src, torch.Tensor):
        points_src = points_src.detach().cpu().numpy()
    points_src = np.asarray(points_src, dtype=float)
    if points_src.ndim == 1:
        if points_src.shape[0] != 2:
            raise ValueError(f"points_src should be 1D with shape (2, ), got {points_src.shape}")
        points_src = points_src.reshape(1, 2)  # Reshape to (1, 2) for consistency
    elif points_src.ndim != 2 or points_src.shape[1] != 2:
        raise ValueError(f"points_src should be 2D with shape (N, 2), got {points_src.shape}")
    if Normalize:
        src_norm = points_src / np.array(src_wh, dtype=float)
    else:
        src_norm = points_src
    src_h = np.hstack([src_norm, np.ones((src_norm.shape[0],1))])
    proj = (H_norm @ src_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return proj * np.array(dst_wh, dtype=float)

def _to_1d(a):
    # convert df of  list / tuple / Tensor / ndarray into 1D numpy
    if torch.is_tensor(a):
        a = a.detach().cpu().numpy()
    a = np.array(a)
    return a.reshape(-1)

def parse_target_item(item):
    """Parse a target item into (cx, cy, w, h) format."""
    if isinstance(item, (list, tuple)) and len(item) == 2:
        center, size = _to_1d(item[0]), _to_1d(item[1])
        assert center.size == 2 and size.size == 2, f"Bad item shapes: center={center.shape}, size={size.shape}"
        cx, cy = center.tolist()
        w, h   = size.tolist()
        return float(cx), float(cy), float(w), float(h)

    arr = _to_1d(item)
    if arr.size == 4:
        cx, cy, w, h = arr.tolist()
        return float(cx), float(cy), float(w), float(h)

    raise ValueError(f"Unrecognized target item shape: {arr.shape}, content={arr}")

def load_all_kpts(dir2kpts: str):
    '''Load all keypoints from the specified directory and return as a dictionary.
    Args:
        dir2kpts: Directory containing keypoints files.'''
    src_list, dst_list = [], []
    for filename in sorted(os.listdir(dir2kpts)):
        if filename.endswith('.npz'):
            path = os.path.join(dir2kpts, filename)
            data = np.load(path)
            if 'src' not in data or 'dst' not in data:
                raise ValueError(f"File {filename} does not contain 'src' and 'dst' keys.")
            points_src = data['src']
            points_dst = data['dst']
            if points_src.ndim == 1:
                points_src = points_src.reshape(1, -1)
            if points_dst.ndim == 1:
                points_dst = points_dst.reshape(1, -1)
            src_list.append(points_src)
            dst_list.append(points_dst)

            print(f"[info] Loaded {filename}, src={points_src.shape}, dst={points_dst.shape}")
    points_src_all = np.vstack(src_list)
    points_dst_all = np.vstack(dst_list)
    return points_src_all, points_dst_all

# def get_labels_from_yolo_results(yolo_results_dir: str, target_length, event_start_frame = 114):
#     # Extract labels from YOLO results
#     target_tensors = np.empty(target_length, dtype=object) # start from an empty array and would be turned into a tensor

#     for filename in sorted(filter(lambda f: f.endswith('.txt'), os.listdir(yolo_results_dir))):
#         with open(os.path.join(yolo_results_dir, filename)) as file:
#             target_data = [float(value)
#                             for line in file
#                             if int(line.strip().split()[0]) in {2, 5, 7}  # Filter for target classes
#                             for value in line.strip().split()[1:5] # get x, y, width, height
#                             ]
#             target_tensor = torch.tensor(target_data, dtype=torch.float).view(-1, 4) # Reshape to (N, 4)
#             index = int(filename[3:-4])
#             if 0 <= index < target_length:
#                 target_tensors[index] = target_tensor
#     return target_tensors # Return as tensor or empty tensor, here can not stack as the n_objects on each frames is not the same

def nearest_neighbor_matching(video_fps=90, event_fps=100, clip_length=60, video_start_frame=114, event_start_frame=177, include_last_frame=False, n_video_frames=None):
    """Match video frames to event frames based on nearest neighbor in time."""
    if n_video_frames is None:
        n_video_frames = int(round(video_fps * clip_length))
        if not include_last_frame:
            n_video_frames = max(0, n_video_frames - 1)  # Exclude the last frame if not included
    if n_event_frames is None:
        n_event_frames = int(round(event_fps * clip_length))
        if not include_last_frame:
            n_event_frames = max(0, n_event_frames - 1)
    
    v_idx = np.arange(video_start_frame, n_video_frames, dtype=int)
    e_idx = np.arange(event_start_frame, n_event_frames, dtype=int)

    if v_idx.size == 0 or e_idx.size == 0:
        return {
            "video_indices": v_idx,
            "event_indices": e_idx,
            "video_to_event": np.array([], dtype=int),
            "event_to_video": np.array([], dtype=int),
        }
    
    v_t = (v_idx - video_start_frame) / float(video_fps) 
    e_t = (e_idx - event_start_frame) / float(event_fps)

    v2e = event_start_frame + np.rint(v_t * event_fps).astype(int)
    v2e = np.clip(v2e, 0, n_event_frames - 1)

    e2v = video_start_frame + np.rint(e_t * video_fps).astype(int)
    e2v = np.clip(e2v, 0, n_video_frames - 1)

    return {
        "video_indices": v_idx,
        "event_indices": e_idx,
        "video_to_event": v2e,
        "event_to_video": e2v,
    }

def count_events(frame, min_max):
    y_min, y_max, x_min, x_max = min_max
    number_of_events = torch.sum(frame > 0)
    min_events_threshold = max(1, 0.01 * ((x_max-x_min)*(y_max-y_min)))
    if number_of_events < min_events_threshold:
        ratio = 0
    else:
        ratio = (number_of_events - min_events_threshold) / ((x_max-x_min)*(y_max-y_min))
    return float(ratio)

def leaky_integrator(input_data, tau, v):
    v_next = v + (1/tau) * (input_data - v)
    return v_next

def crop_video_clip(input_video_path, output_video_path, top_left, crop_size, max_frames=None):
    """
    Crop a region from a video clip.

    :param input_video_path: Path to input video
    :param output_video_path: Path to save cropped video
    :param top_left: (x, y) top-left corner of the crop
    :param crop_size: (width, height) of the crop
    :param max_frames: optional number of frames to limit processing
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width_crop, height_crop = crop_size
    x, y = top_left

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width_crop, height_crop))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Crop the frame
        cropped = frame[y:y+height_crop, x:x+width_crop]
        out.write(cropped)

        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            break

    cap.release()
    out.release()
    return f"Cropped video saved to: {output_video_path}"