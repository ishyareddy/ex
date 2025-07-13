import cv2
import numpy as np
from utils import compute_direction

def extract_movement_sequence(video_path, roi):
    """
    Given a video and ROI, returns the sequence of directional movement angles
    and the video FPS.
    """
    cap = cv2.VideoCapture(video_path)
    x, y, w, h = roi
    fps = cap.get(cv2.CAP_PROP_FPS)

    prev_center = None
    angles = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi_frame = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                curr_center = (cx, cy)

                if prev_center is not None:
                    angle = compute_direction(prev_center, curr_center)
                    angles.append(angle)

                prev_center = curr_center

    cap.release()
    return angles, fps
