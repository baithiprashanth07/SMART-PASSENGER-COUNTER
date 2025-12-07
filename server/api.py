import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, Response, jsonify, render_template, request
import threading
import cv2
import time
import numpy as np
from collections import defaultdict

# Import pipeline components
# You may need to ensure these files exist and are correct; based on file list they do.
from models.detection_pipeline import DetectionPipeline
from core.passenger_counter import PassengerCounter
from core.logger import Logger
from server.websocket_manager import socketio, send_realtime_update, send_detection_update, send_fps_update

app = Flask(__name__)
# Initialize SocketIO with this app
socketio.init_app(app)

# Global variables
output_frame = None
lock = threading.Lock()

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
cap = None
from core.input_reader import InputReader
try:
    # Force live camera or use config default
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
    fps_start_time = cv2.getTickCount()
    
    while True:
        try:
            # Get frame from webcam or generate demo frame
            if use_demo_mode:
                # Generate a demo frame (640x480 with text)
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

            # WebSocket Updates - Send periodically
            if frame_count % 5 == 0: 
                send_realtime_update({
                    "IN": analytics_data["IN"], 
                    "OUT": analytics_data["OUT"], 
                    "total": analytics_data["total"]
                })
                # Send detections count if needed
                send_detection_update(len(boxes))

            # FPS Calculation
            if frame_count % 30 == 0:
                fps_end_time = cv2.getTickCount()
                time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
                if time_diff > 0:
                    fps = 30 / time_diff
                    send_fps_update(fps)
                fps_start_time = cv2.getTickCount()

            # Draw bounding boxes and info for video stream
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
            
            frame_count += 1
                
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
                time.sleep(0.01)
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    # Serve the dashboard HTML
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/analytics')
def analytics():
    """Return real-time passenger analytics"""
    return jsonify(analytics_data)

@app.route('/api/reset', methods=['POST'])
def reset():
    global analytics_data
    analytics_data = defaultdict(int)
    # Reset internal counters if method exists
    if hasattr(counter, 'reset'):
        counter.reset()
    return jsonify({"status": "success", "message": "Counters reset"})

@app.route('/api/change_source', methods=['POST'])
def change_source():
    # Placeholder for change source logic
    try:
        data = request.json
        source = data.get('source')
        print(f"Requested source change to: {source} (Not implemented fully)")
        return jsonify({"status": "success", "source": source})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/status')
def status():
    return jsonify({
        "status": "online",
        "uptime": "TODO", 
        "fps": 0 # We could track global fps
    })

if __name__ == "__main__":
    # Start frame capture thread
    t = threading.Thread(target=capture_frames, daemon=True)
    t.start()

    # Start Flask-SocketIO server
    # Note: socketio.run handles the web server
    print("🚀 Starting Smart Passenger Counter Server on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)