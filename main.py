import os
import cv2
import json
import pandas as pd
from datetime import datetime
from movement_analyser import extract_movement_sequence
from feedback_manager import FeedbackManager
from utils import select_roi, extract_clip
from scipy.spatial.distance import cosine
import numpy as np

REF_VIDEO = "ref2.avi"
TEST_VIDEO = "test3.avi"
MEMORY_FILE = "memory/pattern_memory.json"
LOG_FILE = "logs/deviation_log.xlsx"
CLIP_DIR = "clips"

def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("memory", exist_ok=True)
    os.makedirs(CLIP_DIR, exist_ok=True)

    print("Select a ROI and then press SPACE or ENTER button!")
    print("Cancel the selection process by pressing c button!")

    roi = select_roi(TEST_VIDEO)

    # Extract reference pattern
    ref_angles, fps = extract_movement_sequence(REF_VIDEO, roi)

    feedback = FeedbackManager(MEMORY_FILE)

    # Load previous logs if exists
    if os.path.exists(LOG_FILE):
        log_df = pd.read_excel(LOG_FILE)
        log = log_df.to_dict(orient='records')
    else:
        log = []

    # Extract test video movement patterns
    test_angles, _ = extract_movement_sequence(TEST_VIDEO, roi)

    # Segment into cycles based on reference pattern length
    total_frames = len(test_angles)
    cycle_length = len(ref_angles)
    cycles = total_frames // cycle_length
    step = cycle_length

    print(f"🔷 Total test video frames: {total_frames}")
    print(f"🟩 Total detected cycles: {cycles}")


    for i in range(cycles):
        start = i * step
        end = (i + 1) * step
        angles = test_angles[start:end]

        test = np.array(angles)
        min_len = min(len(angles), len(ref_angles))
        test = np.array(angles[:min_len])
        ref = np.array(ref_angles[:min_len])

        distance = cosine(test, ref) if len(test) > 1 else 1.0
        print(f"Cycle {i+1}: similarity distance → {distance:.4f}")

        timestamp = datetime.now().isoformat()
        cycle_id = i + 1
        clip_filename = f"cycle_{cycle_id}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.mp4"
        clip_path = os.path.join(CLIP_DIR, clip_filename)

        # ✅ Auto-accept if identical or very close to reference
        if distance < feedback.similarity_threshold:
            print(f"✅ Cycle {cycle_id} auto-accepted (close to reference)")
            feedback.add_normal(angles)
            continue

        # Memory checks using similarity
        if feedback.is_known_normal(angles):
            print(f"✅ Cycle {cycle_id} auto-accepted (known normal)")
            continue
        elif feedback.is_known_deviation(angles):
            print(f"❌ Cycle {cycle_id} auto-flagged (known deviation)")
            print(f"ℹ️ Previously confirmed deviation — logging without asking.")
            extract_clip(TEST_VIDEO, start, end, roi, clip_path, fps)
            log.append({"cycle": cycle_id, "status": "Deviation", "timestamp": timestamp, "clip": clip_filename})
            continue

        # Unknown → ask user after preview
        extract_clip(TEST_VIDEO, start, end, roi, clip_path, fps)

        cap_preview = cv2.VideoCapture(clip_path)
        while cap_preview.isOpened():
            ret, frame = cap_preview.read()
            if not ret:
                break
            cv2.imshow(f"Preview: Cycle {cycle_id}", frame)
            if cv2.waitKey(int(1000 // fps)) & 0xFF == ord('q'):
                break
        cap_preview.release()
        cv2.destroyWindow(f"Preview: Cycle {cycle_id}")

        response = input(f"Is this a valid deviation? (y/n): ").strip().lower()
        if response == "y":
            feedback.add_deviation(angles)
            print("✅ Deviation logged.")
            log.append({"cycle": cycle_id, "status": "Deviation", "timestamp": timestamp, "clip": clip_filename})
        else:
            feedback.add_normal(angles)
            print("ℹ️ Pattern added to memory as known-normal.")
            # (normal not logged anymore as per last requirement)

    # Save cumulative logs and memory
    pd.DataFrame(log).to_excel(LOG_FILE, index=False)
    feedback.save_memory()
    print(f"📝 Log updated in {LOG_FILE}")

if __name__ == "__main__":
    main()
