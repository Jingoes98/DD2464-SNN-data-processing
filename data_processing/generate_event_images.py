from utility import *
clip_id = 4
dir2event_frames = "raw_data/2024_09_10_16_00_41_recordings/processed/event_frames"
path2event_frames = os.path.join(dir2event_frames, f"event_frames_{clip_id}.pt")
output_dir = "raw_data/2024_09_10_16_00_41_recordings/processed/event_images"
event_frames = torch.load(path2event_frames)
print(f"Loaded event frames from: {path2event_frames}")
event_frames_to_imgs(event_frames=event_frames, output_dir=output_dir, n_frames=2000)
