import numpy as np
from core.yolo_pipeline import YOLOPipeline
from core.reid_optimized import ReIDPipeline


class DetectionPipeline:
    """
    Unified detection pipeline combining:
    - YOLO person detection
    - Optional ReID for unique person identification
    """

    def __init__(
        self,
        yolo_model="models/yolov8n.onnx",
        reid_model="models/face_recognition.onnx",
        conf_threshold=0.4,
        use_reid=False,
        device="cpu"
    ):
        self.conf_threshold = conf_threshold
        self.use_reid = use_reid
        
        # Initialize YOLO detector
        self.yolo = YOLOPipeline(
            model_path=yolo_model,
            device=device,
            conf_th=conf_threshold,
            nms_th=0.45
        )
        
        # Initialize ReID if enabled
        self.reid = None
        if use_reid:
            try:
                self.reid = ReIDPipeline(
                    model_path=reid_model,
                    similarity_th=0.6,
                    every_n_frames=5
                )
            except Exception as e:
                print(f"⚠️  ReID initialization failed: {e}")
                print("Continuing without ReID...")
                self.use_reid = False
        
        self.frame_count = 0

    def process_frame(self, frame):
        """
        Process a single frame and return detections.
        
        Returns:
            List of detections, each containing:
            {
                "box": [x1, y1, x2, y2],
                "confidence": float,
                "class": int,
                "embedding": np.array (if ReID enabled)
            }
        """
        self.frame_count += 1
        
        # Run YOLO detection
        detections = self.yolo.infer(frame)
        
        results = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            
            # Filter for person class (class 0 in COCO)
            if int(cls) != 0:
                continue
            
            detection_dict = {
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(conf),
                "class": int(cls),
                "embedding": None
            }
            
            # Extract ReID embedding if enabled
            if self.use_reid and self.reid:
                try:
                    is_unique, emb = self.reid.process(
                        frame,
                        track_id=len(results),
                        person_bbox=[x1, y1, x2, y2],
                        frame_count=self.frame_count
                    )
                    detection_dict["embedding"] = emb
                    detection_dict["is_unique"] = is_unique
                except Exception as e:
                    # Silently continue if ReID fails
                    pass
            
            results.append(detection_dict)
        
        return results
