# Smart Passenger Counter 🚌

Real-time passenger counting system using YOLO detection, SORT tracking, and optional face recognition for accurate IN/OUT counting.

## Features

- ✅ **Ultra-fast YOLO ONNX inference** with CPU/GPU support
- ✅ **SORT tracking** with Kalman filter for robust object tracking
- ✅ **Multi-line counting** with directional IN/OUT logic
- ✅ **Optional ReID** for unique person identification
- ✅ **Enhanced Web Dashboard** with real-time Chart.js analytics and modern UI
- ✅ **Live streaming** via Flask web server
- ✅ **Real-time analytics** with WebSocket updates
- ✅ **Auto-tuning** for optimal parameters
- ✅ **Threaded video reader** with RTSP auto-reconnect

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download ONNX Models

**Option A: YOLOv8 (Recommended)**
```bash
# Export YOLOv8n to ONNX
pip install ultralytics
yolo export model=yolov8n.pt format=onnx
# Move to models/yolov8n.onnx
```

**Option B: Use Pre-trained Models**
- Download YOLOv8n ONNX from [Ultralytics](https://github.com/ultralytics/ultralytics)
- Place in `models/yolov8n.onnx`

**Face Recognition (Optional)**
- Download ArcFace or FaceNet ONNX model
- Place in `models/face_recognition.onnx`

### 3. Generate Configuration

```bash
python config/config_generator.py
```

This creates `config/config.yaml` with default settings.

## Quick Start

### Basic Usage (Webcam)

```bash
python main.py
```

### With Video File

Edit `config/config.yaml`:
```yaml
input:
  source: "path/to/video.mp4"
```

Then run:
```bash
python main.py
```

### With RTSP Stream

Edit `config/config.yaml`:
```yaml
input:
  source: "rtsp://your_camera_ip:554/stream"
  reconnect: true
```

## Web Server (Enhanced Dashboard)

Start the Flask server for live streaming and the enhanced real-time dashboard:

```bash
python server/api.py
```

Access:
- **Enhanced Dashboard**: http://localhost:5000/ (Features live video, real-time charts, and controls)
- **Video Stream**: http://localhost:5000/video_feed
- **Analytics API**: http://localhost:5000/api/analytics

## Configuration

Edit `config/config.yaml` to customize:

### Video Input
```yaml
input:
  source: 0  # Webcam, file path, or RTSP URL
  buffer_size: 1
  reconnect: true
```

### Detection & Tracking
```yaml
yolo:
  conf: 0.4  # Detection confidence threshold
  device: cpu  # or "cuda" for GPU

tracking:
  max_age: 30  # Max frames to keep track
  min_hits: 3  # Min detections before confirmed
```

### Counting Lines
```yaml
counting:
  lines:
  - name: entry
    coords: [0, 360, 1280, 360]  # [x1, y1, x2, y2]
    direction: vertical  # Cross direction
```

## Auto-Tuning

Automatically find optimal parameters:

```bash
python tuning/auto_tune.py
```

This will:
1. Test different parameter combinations
2. Evaluate tracking quality and FPS
3. Save best config to `config/config_optimized.yaml`

## Project Structure

```
smart_passenger_counter/
├── config/
│   ├── config.yaml                 ← Runtime settings
│   └── config_generator.py         ← Config generator
├── models/
│   ├── yolov8n.onnx                ← YOLO model
│   └── face_recognition.onnx       ← ReID model (optional)
├── core/
│   ├── yolo_pipeline.py            ← YOLO inference
│   ├── sort_fast.py                ← SORT tracker
│   ├── directional_counter_multi.py← Multi-door counting
│   ├── reid_optimized.py           ← Face recognition
│   ├── detection_tracker.py        ← YOLO+SORT+ReID fusion
│   ├── passenger_counter.py        ← Simple line counter
│   ├── input_reader.py             ← Threaded video reader
│   ├── logger.py                   ← CSV/JSON logging
│   └── utils.py                    ← Helper functions
├── server/
│   ├── templates/
│   │   ├── index.html              ← Enhanced Dashboard UI
│   │   └── dashboard.js            ← Enhanced Chart.js/WebSocket logic
│   ├── api.py                      ← Enhanced Flask streaming server & API
│   └── websocket_manager.py        ← Enhanced Real-time updates
├── tuning/
│   ├── auto_tune.py                ← Parameter tuning
│   └── metrics.py                  ← Accuracy evaluation
├── main.py                         ← Main runner
└── README.md                       ← This file
```

## API Endpoints

The web server now exposes enhanced API endpoints:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Returns the **Enhanced Dashboard** HTML page. |
| `/video_feed` | `GET` | Live MJPEG video stream with annotations. |
| `/api/analytics` | `GET` | Returns current passenger counts and performance metrics. |
| `/api/reset` | `POST` | Resets all passenger counters to zero. |
| `/api/change_source` | `POST` | Changes the video input source (webcam, RTSP, file). |
| `/api/status` | `GET` | Returns system status information (uptime, FPS, source). |

## Logging

Logs are saved to `logs/passenger_log.csv` by default:

```csv
timestamp,frame_count,detections,in_count,out_count,total_count
2024-12-04T22:00:00,1,3,1,0,1
2024-12-04T22:00:01,2,2,0,0,1
```

## Performance Tips

1. **Use GPU**: Set `device: cuda` in config for 5-10x speedup
2. **Lower resolution**: Reduce `input_size` to 416 or 320
3. **Disable ReID**: Set `reid.enabled: false` if not needed
4. **Adjust buffer**: Increase `buffer_size` for smoother streaming
5. **Run auto-tune**: Find optimal parameters for your setup

## Troubleshooting

### No detections
- Lower `yolo.conf` threshold (try 0.3)
- Check if ONNX model is valid
- Verify video source is working

### Poor tracking
- Run auto-tuning to optimize parameters
- Increase `tracking.min_hits` for more stable tracks
- Adjust `tracking.iou_threshold`

### RTSP connection issues
- Enable `input.reconnect: true`
- Check network connectivity
- Verify RTSP URL format

## License

MIT License - See LICENSE file for details

## Credits

- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)
- SORT: [Alex Bewley](https://github.com/abewley/sort)
- ArcFace: [DeepInsight](https://github.com/deepinsight/insightface)
- **Enhanced Dashboard**: Implemented by Manus AI with Chart.js and Flask-SocketIO.
