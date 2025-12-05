import numpy as np
from core.yolo_pipeline import YOLOPipeline
from core.sort_fast import Sort
from core.reid_optimized import ReIDPipeline


class TrackedObject:
    """Container for tracked object data."""
    def __init__(self, track_id, bbox, confidence=1.0):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.embedding = None
        self.door_event = None  # "enter" or "exit"


class DetectionTracker:
    """
    Fusion of YOLO + SORT + ReID for robust tracking.
    Combines:
    - YOLO for person detection
    - SORT for tracking
    - Optional ReID for unique identification
    """

    def __init__(self, config=None):
        """
        Args:
            config: Configuration dictionary with:
                - yolo: YOLO config (model_path, conf, iou)
                - tracking: SORT config (max_age, min_hits, iou_threshold)
                - reid: ReID config (model_path, similarity, every_n_frames)
        """
        if config is None:
            config = self._default_config()
        
        # Initialize YOLO detector
        yolo_cfg = config.get("yolo", {})
        self.yolo = YOLOPipeline(
            model_path=yolo_cfg.get("model_path", "models/yolov8n.onnx"),
            device=yolo_cfg.get("device", "cpu"),
            conf_th=yolo_cfg.get("conf", 0.4),
            nms_th=yolo_cfg.get("iou", 0.45)
        )
        
        # Initialize SORT tracker
        tracking_cfg = config.get("tracking", {})
        self.tracker = Sort(
            max_age=tracking_cfg.get("max_age", 30),
            min_hits=tracking_cfg.get("min_hits", 3),
            iou_threshold=tracking_cfg.get("iou_threshold", 0.3)
        )
        
        # Initialize ReID (optional)
        self.reid = None
        reid_cfg = config.get("reid", {})
        if reid_cfg.get("enabled", False):
            try:
                self.reid = ReIDPipeline(
                    model_path=reid_cfg.get("model_path", "models/face_recognition.onnx"),
                    similarity_th=reid_cfg.get("similarity", 0.6),
                    every_n_frames=reid_cfg.get("every_n_frames", 5)
                )
            except Exception as e:
                print(f"⚠️  ReID initialization failed: {e}")
                self.reid = None
        
        self.frame_count = 0

    def update(self, frame):
        """
        Update tracker with new frame.
        
        Args:
            frame: Input frame (BGR image)
        
        Returns:
            Dictionary of tracked objects: {track_id: TrackedObject}
        """
        self.frame_count += 1
        
        # 1. YOLO Detection
        detections = self.yolo.infer(frame)
        
        # Filter for person class (class 0)
        person_dets = []
        for det in detections:
            if int(det[5]) == 0:  # Person class
                person_dets.append(det)
        
        # Convert to numpy array for SORT
        if len(person_dets) > 0:
            dets_array = np.array(person_dets)
        else:
            dets_array = np.empty((0, 5))
        
        # 2. SORT Tracking
        tracked = self.tracker.update(dets_array)
        
        # 3. Create TrackedObject instances
        tracked_objects = {}
        for trk in tracked:
            x1, y1, x2, y2, track_id = trk
            track_id = int(track_id)
            
            obj = TrackedObject(
                track_id=track_id,
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                confidence=1.0
            )
            
            # 4. Optional ReID
            if self.reid:
                try:
                    is_unique, emb = self.reid.process(
                        frame,
                        track_id=track_id,
                        person_bbox=[x1, y1, x2, y2],
                        frame_count=self.frame_count
                    )
                    obj.embedding = emb
                except Exception:
                    pass
            
            tracked_objects[track_id] = obj
        
        return tracked_objects

    @staticmethod
    def _default_config():
        """Return default configuration."""
        return {
            "yolo": {
                "model_path": "models/yolov8n.onnx",
                "device": "cpu",
                "conf": 0.4,
                "iou": 0.45
            },
            "tracking": {
                "max_age": 30,
                "min_hits": 3,
                "iou_threshold": 0.3
            },
            "reid": {
                "enabled": False,
                "model_path": "models/face_recognition.onnx",
                "similarity": 0.6,
                "every_n_frames": 5
            }
        }
