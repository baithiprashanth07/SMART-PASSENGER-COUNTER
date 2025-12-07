import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, Response, jsonify, request, render_template

from websocket_manager import socketio, send_realtime_update
from core.input_reader import InputReader
from core.detection_tracker import DetectionTracker
from core.directional_counter_multi import MultiDoorCounter
import threading
import time
import cv2
import yaml

app = Flask(__name__)
socketio.init_app(app)

# Global shared state
global_frame = None
lock = threading.Lock()

# Load config
CONFIG_PATH = "config/config.yaml"
config = yaml.safe_load(open(CONFIG_PATH, "r"))

# Initialize modules
reader = InputReader(config["input"])
detector = DetectionTracker(config)
counter = MultiDoorCounter(config["counting"]["doors"])

# ------------------------------------------------------
# Thread: Processing pipeline
# ------------------------------------------------------
def processing_loop():
    global global_frame

    while True:
        frame = reader.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        tracked_objects = detector.update(frame)
        door_counts = counter.update(tracked_objects)

        # Drawing for display
        for tid, obj in tracked_objects.items():
            x1, y1, x2, y2 = map(int, obj.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Draw door lines
        for dname, dcfg in config["counting"]["doors"].items():
            Ax1, Ay1, Ax2, Ay2 = dcfg["line_A"]
            Bx1, By1, Bx2, By2 = dcfg["line_B"]
            cv2.line(frame, (Ax1, Ay1), (Ax2, Ay2), (255,255,0), 2)
            cv2.line(frame, (Bx1, By1), (Bx2, By2), (0,128,255), 2)

        with lock:
            global_frame = frame.copy()

        # Send updates to dashboard
        send_realtime_update(door_counts)

# Start background thread
threading.Thread(target=processing_loop, daemon=True).start()

# ------------------------------------------------------
# MJPEG Video Stream
# ------------------------------------------------------
def generate_mjpeg():
    global global_frame

    while True:
        with lock:
            if global_frame is None:
                continue
            ret, buffer = cv2.imencode(".jpg", global_frame)
            frame_data = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_data +
            b"\r\n"
        )
        time.sleep(0.03)  # ~30 FPS


@app.route("/video")
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ------------------------------------------------------
# REST API ENDPOINTS
# ------------------------------------------------------
@app.route("/api/stats")
def get_stats():
    return jsonify(counter.counts)

@app.route("/api/reset", methods=["POST"])
def reset_counters():
    for d in counter.counts:
        counter.counts[d]["enter"] = 0
        counter.counts[d]["exit"] = 0
        counter.counts[d]["occupancy"] = 0
    return jsonify({"status": "ok", "message": "Counters reset"})

@app.route("/api/change_source", methods=["POST"])
def change_source():
    data = request.json
    new_src = data.get("source")

    reader.change_source(new_src)

    return jsonify({"status": "ok", "new_source": new_src})

# ------------------------------------------------------
# DASHBOARD WEBPAGE
# ------------------------------------------------------
@app.route("/")
def dashboard():
    return render_template("index.html")


if __name__ == "__main__":
    print("[✔] Starting Flask server on http://0.0.0.0:5000/")
    socketio.run(app, host=config["server"]["host"], port=config["server"]["port"], debug=False)
