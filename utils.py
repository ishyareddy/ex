import cv2
import os
import numpy as np

def compute_direction(p1, p2):
    """
    Compute the angle of movement from point p1 to p2.
    Returns angle in radians.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.arctan2(dy, dx)

def select_roi(video_path, max_width=1280, max_height=720):
    """
    Let the user select a region of interest (ROI) from the first frame.
    Scales the video to fit the screen for easier selection.
    Returns (x, y, w, h).
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(" Cannot read video for ROI selection.")
    orig_h, orig_w = frame.shape[:2]
    scale_w = max_width / orig_w
    scale_h = max_height / orig_h
    scale = min(scale_w, scale_h, 1.0)

    display_frame = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)))
    roi = cv2.selectROI("Select ROI", display_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    x, y, w, h = roi
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)
    cap.release()
    return (x, y, w, h)

def extract_clip(video_path, start_frame, end_frame, roi, output_path, fps):
    """
    Extract a cropped video clip from the given frame range and ROI.
    Saves it as an MP4 to the given output path.
    """
    cap = cv2.VideoCapture(video_path)
    x, y, w, h = roi
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    while frame_idx < (end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[y:y+h, x:x+w]
        out.write(crop)
        frame_idx += 1

    cap.release()
    out.release()

def trim_or_pad(arr, length):
    """
    Trim or pad a list to a fixed length.
    Pads with 0s if shorter, trims if longer.
    """
    if len(arr) > length:
        return arr[:length]
    return arr + [0] * (length - len(arr))
