from utility import *
clip_id = 3
clip_path = f"raw_data/2024_09_10_16_00_41_recordings/processed/cropped_video_clips/cropped_clip_{clip_id}.mp4"
# clip_path = f"raw_data/2024_09_10_16_00_41_recordings/processed/video_clips/clip_{clip_id}.mp4"

cap = cv2.VideoCapture(clip_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}")
ret, frame = cap.read()
if ret:
    height, width = frame.shape[:2]
    print(f"First frame resolution: width={width}, height={height}")
cap.release()
run_yolo_on_clip(clip_path, clip_id, output_dir="raw_data/2024_09_10_16_00_41_recordings/processed", cropped=True)
