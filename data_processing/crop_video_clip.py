from utility import *

# also a helper script to calculate crop box for event frames
clip_id = 3
input_video_path = f"raw_data/2024_09_10_16_00_41_recordings/processed/video_clips/clip_{clip_id}.mp4"
output_video_dir = f"raw_data/2024_09_10_16_00_41_recordings/processed/cropped_video_clips"
os.makedirs(output_video_dir, exist_ok=True)
output_video_path = os.path.join(output_video_dir, f"cropped_clip_{clip_id}.mp4")
top_left = (580, 530)  # (x, y) coordinates of the top-left corner of the crop box
crop_size = (640, 640)  # (width, height) of the crop box
crop_video_clip(
    input_video_path=input_video_path,           # 原始 video clip 路径
    output_video_path=output_video_path,  # 保存裁剪后视频路径
    top_left=top_left,                                   # 裁剪框左上角坐标 (x, y)
    crop_size=crop_size,                                  # 裁剪框尺寸 (width, height)                                      # （可选）最多处理前 1000 帧
)

# Convert the top-left corner and crop size to event frame coordinates
video_w, video_h = 1920, 1200
event_w, event_h = 640, 480
x0, y0 = top_left
crop_w, crop_h = crop_size

event_x0 = int(x0 * (event_w / video_w))
event_y0 = int(y0 * (event_h / video_h))
event_crop_w = int(crop_w * (event_w / video_w))
event_crop_h = int(crop_h * (event_h / video_h))
print(f"CROP_BOX for event frame:: top_left=({event_x0}, {event_y0}), crop_size=({event_crop_w}, {event_crop_h})")