import numpy as np
import cv2
import onnxruntime as ort
import time


# -------------------------------------------------------
# YOLO-FACE Detector (ONNX)
# -------------------------------------------------------
class YOLOFaceDetector:
    def __init__(self, model_path="models/yoloface.onnx", device="cuda"):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def detect_face(self, frame, person_bbox):
        x1, y1, x2, y2 = map(int, person_bbox)
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return None, None

        img = cv2.resize(roi, (640, 640))
        blob = img.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, :]

        preds = self.session.run(None, {self.input_name: blob})[0]

        # If no face found
        if len(preds) == 0:
            return None, None

        # Best face detection (only 1)
        det = preds[0]
        fx1, fy1, fx2, fy2, conf = det[:5]

        if conf < 0.5:
            return None, None

        # Convert coords back to original person ROI
        fx1 = int((fx1 / 640) * roi.shape[1]) + x1
        fy1 = int((fy1 / 640) * roi.shape[0]) + y1
        fx2 = int((fx2 / 640) * roi.shape[1]) + x2
        fy2 = int((fy2 / 640) * roi.shape[0]) + y2

        face = frame[fy1:fy2, fx1:fx2]

        if face.size == 0:
            return None, None

        face = cv2.resize(face, (112, 112))
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        blob = face_rgb.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, :]

        return blob, [fx1, fy1, fx2, fy2]


# -------------------------------------------------------
# ArcFace (ONNX) embedding extractor
# -------------------------------------------------------
class ArcFaceEmbedder:
    def __init__(self, model_path="models/arcface.onnx", device="cuda", fp16=True):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.fp16 = fp16

    def get_embedding(self, face_blob):
        if face_blob is None:
            return None

        if self.fp16:
            face_blob = face_blob.astype(np.float16)

        output = self.session.run(None, {self.input_name: face_blob})[0]
        emb = output[0]
        emb = emb / (np.linalg.norm(emb) + 1e-6)
        return emb


# -------------------------------------------------------
# ReID Pipeline
# -------------------------------------------------------
class ReIDPipeline:
    def __init__(self, model_path, similarity_th=0.6, every_n_frames=5):
        self.device = "cuda"
        self.similarity_th = similarity_th
        self.every_n_frames = every_n_frames

        # Face Det + Embeddings
        self.face_detector = YOLOFaceDetector("models/yoloface.onnx", device=self.device)
        self.embedder = ArcFaceEmbedder(model_path, device=self.device)

        # Store embeddings for uniqueness check
        self.unique_embeddings = []
        self.unique_track_ids = []

    @staticmethod
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)

    def process(self, frame, track_id, person_bbox, frame_count):
        # Run only every N frames
        if frame_count % self.every_n_frames != 0:
            return False, None

        # 1. Face Detection
        face_blob, face_bbox = self.face_detector.detect_face(frame, person_bbox)
        if face_blob is None:
            return False, None

        # 2. Embedding Extraction
        emb = self.embedder.get_embedding(face_blob)
        if emb is None:
            return False, None

        # 3. Match with known embeddings
        if len(self.unique_embeddings) == 0:
            self.unique_embeddings.append(emb)
            self.unique_track_ids.append(track_id)
            return True, emb

        for ref_emb in self.unique_embeddings:
            sim = self.cosine_sim(ref_emb, emb)
            if sim > self.similarity_th:
                return False, emb  # Already seen person

        # New unique person found
        self.unique_embeddings.append(emb)
        self.unique_track_ids.append(track_id)
        return True, emb
