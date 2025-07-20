import json
import cv2

class Detector:
    def __init__(self, ref_img_path, annotations_path):
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)

    def detect(self, image):
        detections = []
        for label, box in self.annotations.items():
            x1, y1, x2, y2 = box
            roi = image[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean_intensity = cv2.mean(gray)[0]
            if mean_intensity > 30:  # Adjust this threshold based on background
                detections.append({
                    "label": label,
                    "box": (x1, y1, x2, y2)
                })
        return detections
