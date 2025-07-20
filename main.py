import cv2
import os
import json
import time
import pandas as pd
from datetime import datetime
from detector import Detector
from layout_matcher import LayoutChecker

VIDEO_PATH = "data/pcb_check_simulation.avi"
REF_IMG_PATH = "data/ref_img.png"
LAYOUT_PATH = "config/layout.json"
MEMORY_FILE = "data/deviation_memory.json"
LOG_FILE = "logs/deviation_log.xlsx"
PREVIEW_DIR = "previews"
CLIP_DURATION_SEC = 2  # Save 2 seconds before & after deviation

os.makedirs(PREVIEW_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Load memory
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r") as f:
        deviation_memory = json.load(f)
else:
    deviation_memory = {}

log_records = []

def draw_detections(frame, detections, check_result):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        color = (0, 255, 0) if check_result["details"].get(label) == "present" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    status_text = f"Status: {check_result['status']}"
    missing_text = ", ".join(check_result["missing"]) if check_result["status"] == "missing" else "No Missing Components"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, missing_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

def save_clip(buffer, timestamp_str, missing):
    out_path = os.path.join(PREVIEW_DIR, f"deviation_{timestamp_str}.mp4")
    height, width = buffer[0].shape[:2]
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
    for frame in buffer:
        out.write(frame)
    out.release()
    print(f"Saved deviation preview: {out_path}")
    return out_path

def log_event(timestamp, status, missing, preview_path):
    log_records.append({
        "timestamp": timestamp,
        "status": status,
        "missing_components": ", ".join(missing),
        "preview_clip": preview_path
    })

def main():
    detector = Detector(ref_img_path=REF_IMG_PATH, annotations_path=LAYOUT_PATH)
    layout_checker = LayoutChecker(LAYOUT_PATH)

    ref_img = cv2.imread(REF_IMG_PATH)
    if ref_img is None:
        print("Error: Could not load reference image.")
        return
    ref_height, ref_width = ref_img.shape[:2]

    print("Reference layout:", list(layout_checker.reference_layout.keys()))

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    buffer = []
    max_buffer_len = int(fps * CLIP_DURATION_SEC)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to grab frame.")
            break

        frame_resized = cv2.resize(frame, (ref_width, ref_height))
        buffer.append(frame_resized.copy())
        if len(buffer) > max_buffer_len:
            buffer.pop(0)

        detections = detector.detect(frame_resized)
        check_result = layout_checker.check(detections)
        missing_key = "_".join(sorted(check_result["missing"]))

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if check_result["status"] == "missing" and missing_key not in deviation_memory:
            # Save clip
            preview_path = save_clip(buffer, timestamp, check_result["missing"])
            deviation_memory[missing_key] = {
                "timestamp": timestamp,
                "components": check_result["missing"]
            }
            log_event(timestamp, check_result["status"], check_result["missing"], preview_path)
        elif check_result["status"] == "missing":
            print("Repeated deviation detected. Skipping clip save.")
            log_event(timestamp, "repeated", check_result["missing"], "already seen")

        draw_detections(frame_resized, detections, check_result)
        cv2.imshow("PCB Component Check", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    # Save memory
    with open(MEMORY_FILE, "w") as f:
        json.dump(deviation_memory, f, indent=2)

    # Save Excel log
    df = pd.DataFrame(log_records)
    df.to_excel(LOG_FILE, index=False)
    print(f"Log saved to {LOG_FILE}")

if __name__ == "__main__":
    main()
