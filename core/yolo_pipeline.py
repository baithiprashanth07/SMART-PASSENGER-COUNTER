import cv2
import numpy as np
import onnxruntime as ort
import time

class YOLOPipeline:
    """
    Ultra-fast YOLO ONNX pipeline supporting:
    - ONNXRuntime CUDA
    - Dynamic shapes
    - Custom NMS
    """

    def __init__(
        self,
        model_path="models/yolov8n.onnx",
        device="cuda",
        conf_th=0.35,
        nms_th=0.45,
        input_size=640,
        fp16=False
    ):
        self.model_path = model_path
        self.device = device
        self.conf_th = conf_th
        self.nms_th = nms_th
        self.input_size = input_size
        self.fp16 = fp16

        # -----------------------------------------
        # Session options for ONNX Runtime
        # -----------------------------------------
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=providers,
            )
        except Exception as e:
            print(f"Failed to create InferenceSession with providers {providers}: {e}")
            print("Falling back to CPUExecutionProvider")
            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [x.name for x in self.session.get_outputs()]

    # ----------------------------------------------------------
    # Letterbox Resize (same as YOLO logic)
    # ----------------------------------------------------------
    def letterbox(self, image, new_shape=640, color=(114, 114, 114)):
        h, w = image.shape[:2]
        scale = min(new_shape / h, new_shape / w)
        nh, nw = int(h * scale), int(w * scale)

        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)

        pad_h = (new_shape - nh) // 2
        pad_w = (new_shape - nw) // 2
        canvas[pad_h:pad_h+nh, pad_w:pad_w+nw] = resized

        return canvas, scale, pad_w, pad_h

    # ----------------------------------------------------------
    # Run YOLO Inference
    # ----------------------------------------------------------
    def infer(self, frame):
        img, scale, pad_w, pad_h = self.letterbox(frame, self.input_size)

        blob = img.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, 0)

        # FP16 acceleration - only if model supports it
        # Standard YOLOv8 export is FP32. 
        # If user explicitly requests FP16 and model expects it, keep it.
        # But for safety with standard export, we default to FP32.
        if self.fp16:
             blob = blob.astype(np.float16)

        outputs = self.session.run(self.output_names, {self.input_name: blob})[0]

        return self.post_process(outputs, frame, scale, pad_w, pad_h)

    # ----------------------------------------------------------
    # Post Processing: Extract boxes + NMS
    # ----------------------------------------------------------
    def post_process(self, preds, original_frame, scale, pad_w, pad_h):
        # preds shape: (1, 84, 8400) for YOLOv8n
        # 84 = 4 (box) + 80 (classes)
        
        preds = preds[0] # (84, 8400)
        
        # Transpose to (8400, 84)
        preds = preds.transpose()
        
        # Extract boxes and scores
        boxes = preds[:, :4]
        scores = preds[:, 4:]
        
        # Get max confidence and class
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence
        mask = confidences > self.conf_th
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []
            
        # Convert boxes from cx, cy, w, h to x1, y1, x2, y2
        # And rescale to original image
        
        final_boxes = []
        for i in range(len(boxes)):
            cx, cy, w, h = boxes[i]
            
            x1 = (cx - w/2 - pad_w) / scale
            y1 = (cy - h/2 - pad_h) / scale
            x2 = (cx + w/2 - pad_w) / scale
            y2 = (cy + h/2 - pad_h) / scale
            
            final_boxes.append([x1, y1, x2, y2, confidences[i], class_ids[i]])
            
        final_boxes = np.array(final_boxes)
        
        # NMS
        final = self.nms(final_boxes, self.nms_th)
        return final

    # ----------------------------------------------------------
    # Non-Max Suppression
    # ----------------------------------------------------------
    @staticmethod
    def nms(boxes, thresh):
        if len(boxes) == 0:
            return []
            
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            idx = np.where(iou <= thresh)[0]
            order = order[idx + 1]

        return boxes[keep]
