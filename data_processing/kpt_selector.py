from utility import *
import skimage.transform

# This is a helper script for selecting kpts pairs in video/event frames to find homography matrix
#====================== Configuration =======================
clip_id = 4 # index of clip
frame_idx = 700 # index of frame
# these settings should adjusted accordingly by your video frame cropping area
video_size = 640 
event_size = 300
CROP_BOX = (180, 180, 480, 480) 
#============================================================

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

dir2event_frames = "raw_data/2024_09_10_16_00_41_recordings/processed/event_frames"
path2event_frames = os.path.join(dir2event_frames, f"event_frames_{clip_id}.pt")
output_dir = "raw_data/2024_09_10_16_00_41_recordings/processed/event_images" # event frames to images
event_frames = torch.load(path2event_frames)
subset, used_idx = extract_event_subset(event_frames, e_idx.tolist()) # extract subset of event frames


frame = subset[frame_idx].to_dense()  # get first event frame
cropped_frame = frame[CROP_BOX[1]:CROP_BOX[3], CROP_BOX[0]:CROP_BOX[2]] 
framen = cropped_frame.float().cpu().numpy()

target_index = v_idx[frame_idx]
print(target_index)
yolo_image = plt.imread(rf"raw_data\2024_09_10_16_00_41_recordings\processed\yolo\result_cropped_{clip_id}\vis\img{target_index:04d}.png") 
selector = PointSelector(yolo_image, framen, max_points=24, id=frame_idx)
plt.show()

dir2kpts = "H_matrix"
homography_points = np.load("homography_points.npz")

points_src = homography_points['src']  # kpts selected on video frames
points_dst = homography_points['dst']  # corresponding points in event frames

H = skimage.transform.estimate_transform('projective', 
                                         points_src/video_size, 
                                         points_dst/event_size)
print("Homography matrix:\n", H.params)
