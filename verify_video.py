import cv2
import os

video_path = "sample_video.mp4"
print(f"Checking {video_path}...")
print(f"Exists: {os.path.exists(video_path)}")
print(f"Size: {os.path.getsize(video_path)} bytes")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Failed to open video")
else:
    print("Video opened successfully")
    ret, frame = cap.read()
    print(f"Read frame: {ret}")
    if ret:
        print(f"Frame shape: {frame.shape}")
    
    # Check FPS and frame count
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"FPS: {fps}")
    print(f"Frame count: {count}")

cap.release()
