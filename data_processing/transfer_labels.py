import os
os.environ['MPLBACKEND'] = 'Agg'
import pandas as pd
import numpy as np
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import cv2
from pathlib import Path
from utility import *
import skimage

#========== CONFIG ===========
# CROP_BOX = (150, 100, 450, 400) # crop_x_min, crop_y_min, crop_x_max, crop_y_max
CROP_BOX = (180, 180, 480, 480) # crop_x_min, crop_y_min, crop_x_max, crop_y_max, make sure this aligns with the event size
torch.cuda.set_device(0)
nr = 0
clip_id = 4
video_size = 640
event_size = 300
output_dir = "raw_data/2024_09_10_16_00_41_recordings/processed"
save_dir = os.path.join(output_dir, f"frames_with_labels/{clip_id}")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

#========== Functions ===========
def annotate_frame(frame, targets, v_idx, i, clip_id, overlay, VISUALIZE=False):
    global max_value
    if VISUALIZE:
        fig, axes = plt.subplots(1, 3, figsize=(20, 20))
        target_idx = v_idx[i]
        yolo_image = plt.imread(rf"raw_data\2024_09_10_16_00_41_recordings\processed\yolo\result_cropped_{clip_id}\vis\img{target_idx:04d}.png") 
        axes[0].imshow(yolo_image)
        x_min, y_min, x_max, y_max = (0, 0, 200, 200)
        axes[1].imshow(frame[y_min:y_max, x_min:x_max], cmap='gray')
    small_frame_dim = 64
    G_overlay = np.zeros((small_frame_dim, small_frame_dim))

    for t in range(len(targets)):
        # center_x, center_y = targets[t][0]
        # w, h = targets[t][1]
        center_x, center_y, w, h = parse_target_item(targets[t])
        x_min = center_x - w/2 
        y_min = center_y - h/2 
        x_max = center_x + w/2
        y_max = center_y + h/2 

        if VISUALIZE:
            rect = patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor='r', facecolor='none')
            cirk = patches.Circle((center_x, center_y), radius=1, edgecolor='r', facecolor='red')
            axes[1].add_patch(rect)    
            axes[1].add_patch(cirk)

        w_small = w * small_frame_dim/200
        h_small  = h * small_frame_dim/200
        center_x_small =  center_x * small_frame_dim/200 #hardcoded adjustment
        center_y_small = center_y * small_frame_dim/200

        sigma_x_small = w_small / 6
        sigma_y_small = h_small / 6
        bbox_region = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
        events_factor = count_events(bbox_region, (y_min, y_max, x_min, x_max))

        # Create Gaussian mask on the smaller frame
        X, Y = np.meshgrid(np.linspace(0, small_frame_dim, small_frame_dim), 
                           np.linspace(0, small_frame_dim, small_frame_dim))
        G = np.exp(-((X - center_x_small)**2 / (2 * sigma_x_small**2) 
                     + (Y - center_y_small)**2 / (2 * sigma_y_small**2)))
        # debug
        G_normalized = G*events_factor
        G_overlay = np.maximum(G_overlay, G_normalized)


    # Apply to the small overlay
    # debug convert from ndarry to tensor
    # G_overlay_tensor = torch.tensor(G_overlay)
    if torch.max(torch.tensor(G_overlay)) > max_value:
        max_value = torch.max(torch.tensor(G_overlay))

    overlay = leaky_integrator(torch.tensor(G_overlay), tau=10, v=torch.tensor(overlay)) 
    label_frame = torch.tensor(overlay, dtype=torch.float32)
    label_frame[label_frame < 1e-4] = 0.0  # Set small values to 0

    if VISUALIZE:
        axes[2].imshow(overlay, cmap='hot', extent=(0, 64, 64, 0), vmin=0, vmax=0.03)
        axes[0].axis('off')
        axes[1].axis('off')
        axes[2].axis('off')
        fig.tight_layout()
        os.makedirs("raw_data/2024_09_10_16_00_41_recordings/processed/frame_with_labels", exist_ok=True)
        fig.savefig(f"raw_data/2024_09_10_16_00_41_recordings/processed/frame_with_labels/{i}.png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    x_min, y_min, x_max, y_max = (0, 0, 200, 200)
    return [torch.tensor(frame[y_min:y_max, x_min:x_max], dtype=torch.float32), label_frame], overlay

#====== Load YOLO Results ======
# target_tensors, mapping = get_labels_from_yolo_results( # yolo results from cropped video
#     yolo_results_dir=f"raw_data/2024_09_10_16_00_41_recordings/processed/yolo/result_cropped_{clip_id}/track/labels",
#     # video_start_frame=114,
#     # event_start_frame=177, # hardcoded for each clip, hardcoded for clip_id 4
#     video_start_frame=0,
#     event_start_frame=65, 
#     target_length=5000,  # Video length is 60 seconds, 90 fps, so 60 * 90 = 5400 frames
#     n_video_frames=5399,  # Adjusted to match the target length
#     return_mapping=True
# )

dir2anchors = "time_alignments/20240910"
path2anchors = os.path.join(dir2anchors, f"{clip_id}.txt")
e_anchors, v_anchors = parse_index_anchors(path2anchors)

e_idx, v_idx = build_time_alignment_index_pairs(
    e_anchors, v_anchors,
    rounding="round",          
)

target_tensors = get_labels_with_indices(
    yolo_results_dir=f"raw_data/2024_09_10_16_00_41_recordings/processed/yolo/result_cropped_{clip_id}/track/labels",
    v_idx=v_idx,
    target_classes={2, 5, 7}
)

# target_tensors, mappings = get_labels_aligned_to_events(
#     yolo_results_dir=f"raw_data/2024_09_10_16_00_41_recordings/processed/yolo/result_cropped_{clip_id}/track/labels",
#     event_start=65, event_count=5000,
#     event_fps=100,
#     video_fps=90,
#     video_start=0,
#     max_event_frames=5999,  # default to 5999 for 60 seconds at 100 fps
#     max_video_frames=5399,  # default to 5399 for 60 seconds at 90 fps
#     target_classes={2, 5, 7},
#     return_mapping=True,
# )

#====== Load Event Frames ======
dir2event_frames = "raw_data/2024_09_10_16_00_41_recordings/processed/event_frames"
path2event_frames = os.path.join(dir2event_frames, f"event_frames_{clip_id}.pt") 
data = torch.load(path2event_frames)
print(f"shape of event frames: {data.shape}")
event_frames, used_idx = extract_event_subset(data, e_idx.tolist()) # uncropped event frames
num_frames = event_frames.shape[0]
print(f"Number of event frames: {num_frames}")

#====== Load Homography Points ======
# homography_points = np.load("homography_points.npz")
# pts_src_px = homography_points['src'] 
# pts_dst_px = homography_points['dst']
dir2kpts = "H_matrix"
pts_src_px, pts_dst_px = load_all_kpts(dir2kpts) 
H_src, W_src = video_size, video_size   # video frame size 
H_dst, W_dst = event_size, event_size  # event frame size
# Normalize points
src = pts_src_px / np.array([W_src, H_src], dtype=float)
dst = pts_dst_px / np.array([W_dst, H_dst], dtype=float)
# H = skimage.transform.estimate_transform('projective', 
#                                          points_src/video_size, 
#                                          points_dst/event_size)
H_norm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
# H = skimage.transform.estimate_transform('euclidean', 
#                                          points_src/video_size, 
#                                          points_dst/event_size)

#======= Batch Processing =======
i = 0
j = 0
frames_tensor = []
label_tensor = []
new_label = []
small_frame_dim = 64
max_value = 0 
overlay = np.zeros((small_frame_dim, small_frame_dim))
video_size = 640
event_size = 300
x_min, y_min, x_max, y_max = CROP_BOX

for frame_index in tqdm(range(num_frames), desc="Processing Event frames"):
    frame = event_frames[frame_index].to_dense()[y_min:y_max, x_min:x_max] # cropped event frame
    warped = []
    
    if target_tensors[j] is None or len(target_tensors[j]) == 0:
        # here we need to mannully crop the frames if we don't run annotate_data()
        print("missing targets for frame", j)
        res = [torch.tensor(frame[0:200, 0:200]), torch.tensor(np.zeros((64, 64)))]
    
    else: 
        for tar in range(len(target_tensors[j])):
            # center_xy = (target_tensors[j][tar][:2] * video_size).numpy()
            center_xy_norm = target_tensors[j][tar][:2].numpy()  # normalized center coordinates``
            # projected = H(center_xy_norm)[0] * event_size
            projected = warp_points_xy(center_xy_norm, H_norm, (W_src, H_src), (W_dst, H_dst), Normalize=False)
            w, h = (target_tensors[frame_index][tar][2:] * event_size).numpy().tolist()
            warped.append([projected.tolist(), [w, h]])
        res, overlay = annotate_frame(frame, warped, v_idx.tolist(), i, clip_id, overlay, VISUALIZE=True)
        # def annotate_frame(frame, targets, mappings, i, clip_id, overlay, VISUALIZE=False):
    
    if i % 500 == 0:
        print(i)

    frames_tensor.append(res[0])
    label_tensor.append(res[1])

    i+=1
    j+=1

torch.save([torch.stack(frames_tensor).to_sparse().coalesce(), torch.stack(label_tensor).to_sparse().coalesce()], f"{save_dir}/processed.pt")
print(f"Saved output to {save_dir}/{clip_id}.pt")