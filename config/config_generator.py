import yaml
from pathlib import Path


def generate_default_config(output_path="config/config.yaml"):
    """
    Generate default configuration file for the passenger counter system.
    """
    
    config = {
        # Video Input Configuration
        "input": {
            "source": 0,  # 0 for webcam, or path to video file, or RTSP URL
            "buffer_size": 1,
            "reconnect": True,  # Auto-reconnect for RTSP streams
            "fps": 30
        },
        
        # YOLO Detection Configuration
        "yolo": {
            "model_path": "models/yolov8n.onnx",
            "device": "cpu",  # "cpu" or "cuda"
            "conf": 0.4,  # Confidence threshold
            "iou": 0.45,  # NMS IOU threshold
            "input_size": 640,
            "fp16": False  # FP16 acceleration (requires CUDA)
        },
        
        # SORT Tracking Configuration
        "tracking": {
            "max_age": 30,  # Max frames to keep track without detection
            "min_hits": 3,  # Min detections before track is confirmed
            "iou_threshold": 0.3  # IOU threshold for matching
        },
        
        # ReID Configuration (Optional)
        "reid": {
            "enabled": False,  # Enable face recognition for unique counting
            "model_path": "models/face_recognition.onnx",
            "similarity": 0.6,  # Similarity threshold
            "every_n_frames": 5  # Process ReID every N frames
        },
        
        # Counting Configuration
        "counting": {
            "mode": "simple",  # "simple" or "multi_door"
            "lines": [
                {
                    "name": "entry",
                    "coords": [0, 360, 1280, 360],  # [x1, y1, x2, y2]
                    "direction": "vertical"  # "vertical" or "horizontal"
                }
            ],
            # Multi-door configuration (if mode is "multi_door")
            "doors": {
                "door1": {
                    "line_A": [200, 300, 800, 300],
                    "line_B": [200, 500, 800, 500]
                }
            }
        },
        
        # Logging Configuration
        "logging": {
            "enabled": True,
            "log_file": "logs/passenger_log.csv",
            "format": "csv",  # "csv" or "json"
            "log_events": True
        },
        
        # Server Configuration
        "server": {
            "host": "0.0.0.0",
            "port": 5000,
            "debug": False,
            "enable_websocket": True
        },
        
        # Display Configuration
        "display": {
            "show_video": True,
            "show_boxes": True,
            "show_lines": True,
            "show_counts": True,
            "window_name": "Passenger Counter"
        }
    }
    
    # Create config directory if it doesn't exist
    config_dir = Path(output_path).parent
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Write config to file
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Generated config file: {output_path}")
    return config


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file."""
    if not Path(config_path).exists():
        print(f"⚠️  Config file not found: {config_path}")
        print("Generating default config...")
        return generate_default_config(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def update_config(config_path, updates):
    """
    Update specific fields in config file.
    
    Args:
        config_path: Path to config file
        updates: Dictionary of updates (nested keys supported)
    """
    config = load_config(config_path)
    
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    config = deep_update(config, updates)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Updated config file: {config_path}")
    return config


if __name__ == "__main__":
    # Generate default configuration
    generate_default_config()
    print("\n📝 Default configuration generated!")
    print("Edit config/config.yaml to customize settings.")
