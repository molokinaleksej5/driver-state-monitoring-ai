from __future__ import annotations

import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.50, margin: float = 0.18):
        self.detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=min_detection_confidence)
        self.margin = margin

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb)
        if not result.detections:
            return None
        h, w = frame.shape[:2]
        det = result.detections[0]
        box = det.location_data.relative_bounding_box
        x1 = int(box.xmin * w)
        y1 = int(box.ymin * h)
        bw = int(box.width * w)
        bh = int(box.height * h)
        pad_x = int(bw * self.margin)
        pad_y = int(bh * self.margin)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x1 + bw + 2 * pad_x)
        y2 = min(h, y1 + bh + 2 * pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def crop(self, frame, bbox):
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size else None
