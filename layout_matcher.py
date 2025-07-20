import json

class LayoutChecker:
    def __init__(self, layout_path):
        with open(layout_path, "r") as f:
            self.reference_layout = json.load(f)

    def check(self, detections):
        detected_labels = set(d["label"] for d in detections)
        layout_labels = set(self.reference_layout.keys())
        missing = list(layout_labels - detected_labels)
        result = {
            "status": "ok" if not missing else "missing",
            "missing": missing,
            "details": {
                label: "present" if label in detected_labels else "missing"
                for label in layout_labels
            }
        }
        return result
