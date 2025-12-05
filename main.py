from models.detection_pipeline import DetectionPipeline
from core.passenger_counter import PassengerCounter
from core.input_reader import InputReader
import cv2
import time
import os

# Initialize unified detection pipeline
# Use CUDA if available, otherwise CPU
pipeline = DetectionPipeline(conf_threshold=0.4, device="cuda")
counter = PassengerCounter()  # Multi-line IN/OUT logic

# Use sample video if available, otherwise default to 0 (webcam)
video_source = "sample_video.mp4" if os.path.exists("sample_video.mp4") else 0
print(f"Using video source: {video_source}")

config = {
    "source": video_source,
    "buffer_size": 2,
    "reconnect": True
}

reader = InputReader(config)

print("Starting main loop...")
prev_time = time.time()
while True:
    frame = reader.get_frame()
    
    if frame is None:
        if reader.stopped:
            break
        time.sleep(0.01)
        continue

    detections = pipeline.process_frame(frame)
    
    # Update passenger counts using bounding boxes
    boxes = [det["box"] for det in detections]
    in_count, out_count = counter.update(boxes)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time > prev_time else 0
    prev_time = curr_time

    # Optional: Draw boxes + lines
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        # Draw confidence and class
        label = f"{det['class']}: {det['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Draw counts
    cv2.putText(frame, f"IN: {in_count} OUT: {out_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display frame
    cv2.imshow("Passenger Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

reader.stop()
cv2.destroyAllWindows()
