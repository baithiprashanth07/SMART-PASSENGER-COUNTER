import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, Response, jsonify
import threading
import cv2
from collections import defaultdict
from models.detection_pipeline import DetectionPipeline
from core.passenger_counter import PassengerCounter
from core.logger import Logger
from core.utils import draw_boxes

app = Flask(__name__)

# Global variables
output_frame = None
lock = threading.Lock()

# Video source (RTSP, webcam, or video file)
video_source = 0  # 0 for webcam, change to video path or RTSP URL
cap = None

# Initialize components with error handling
pipeline = None
counter = PassengerCounter()  # Handles multi-line IN/OUT logic
logger = Logger("logs/passenger_log.csv")

# Try to initialize detection pipeline
try:
    # Use CUDA if available
    pipeline = DetectionPipeline(conf_threshold=0.4, device="cuda")
    print("✅ Detection pipeline initialized")
except Exception as e:
    print(f"⚠️  Detection pipeline failed: {e}")
    print("Running in demo mode without detection...")

# Try to open video capture
from core.input_reader import InputReader
try:
    # Force live camera
    source = 0
    # source = "sample_video.mp4" if os.path.exists("sample_video.mp4") else 0
    
    config = {
        "source": source,
        "buffer_size": 2,
        "reconnect": True
    }
    cap = InputReader(config)
    print(f"✅ Video source opened: {source}")
except Exception as e:
    print(f"⚠️  Video capture error: {e}")
    cap = None

# Real-time analytics
analytics_data = defaultdict(int)


def capture_frames():
    global output_frame, lock, analytics_data
    
    # Check if we have a video source
    use_demo_mode = (cap is None)
    
    if use_demo_mode:
        print("⚠️  No video source available - Running in DEMO MODE with generated frames")
    
    frame_count = 0
    
    while True:
        try:
            # Get frame from webcam or generate demo frame
            if use_demo_mode:
                # Generate a demo frame (640x480 with text)
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:] = (50, 50, 50)  # Dark gray background
                
                # Add demo text
                cv2.putText(frame, "DEMO MODE", (200, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(frame, "No webcam detected", (180, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                cv2.putText(frame, f"Frame: {frame_count}", (250, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                cv2.putText(frame, "Add ONNX models to enable detection", (120, 350), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 100), 1)
                
                frame_count += 1
                import time
                time.sleep(0.033)  # ~30 FPS
            else:
                # Use InputReader
                frame = cap.get_frame()
                
                if frame is None:
                    if cap.stopped:
                        print("⚠️  Video source stopped")
                        use_demo_mode = True
                        continue
                    # Buffer empty, wait a bit
                    import time
                    time.sleep(0.001)
                    continue

            # Detect persons and extract embeddings (if pipeline available)
            boxes = []
            if pipeline is not None and not use_demo_mode:
                try:
                    detections = pipeline.process_frame(frame)
                    boxes = [det["box"] for det in detections]
                except Exception as e:
                    # Detection failed, continue with empty boxes
                    if frame_count % 100 == 0:  # Log occasionally
                        print(f"⚠️  Detection error: {e}")
                    pass

            # Update passenger counting
            in_count, out_count = counter.update(boxes)

            # Update analytics
            analytics_data["IN"] += in_count
            analytics_data["OUT"] += out_count
            analytics_data["total"] = analytics_data["IN"] - analytics_data["OUT"]

            # Draw bounding boxes and info
            annotated_frame = frame.copy()
            if len(boxes) > 0:
                from core.utils import draw_boxes, draw_counting_info
                annotated_frame = draw_boxes(annotated_frame, boxes)
                annotated_frame = draw_counting_info(annotated_frame, counter.get_counts())
            elif not use_demo_mode:
                # Show counts even without detections
                from core.utils import draw_counting_info
                annotated_frame = draw_counting_info(annotated_frame, counter.get_counts())

            # Log detections and counts (only occasionally in demo mode)
            if logger and (not use_demo_mode or frame_count % 30 == 0):
                logger.log({
                    "detections": boxes,
                    "IN": in_count,
                    "OUT": out_count,
                    "total": analytics_data["total"]
                })

            # Update frame for streaming
            with lock:
                output_frame = annotated_frame
                
        except Exception as e:
            print(f"⚠️  Frame capture error: {e}")
            import traceback
            traceback.print_exc()
            continue


def generate_stream():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return jsonify({"message": "Smart Passenger Counting API Running"})


@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/analytics')
def analytics():
    """Return real-time passenger analytics"""
    return jsonify(analytics_data)


if __name__ == "__main__":
    # Start frame capture thread
    t = threading.Thread(target=capture_frames, daemon=True)
    t.start()

    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)
