import cv2
import numpy as np
import os

# --- Output directory ---
output_dir = r"C:\Users\91997\Desktop\Deviation"
os.makedirs(output_dir, exist_ok=True)

ref_path = os.path.join(output_dir, "ref1.avi")
test_path = os.path.join(output_dir, "test3.avi")

# --- Settings ---
width, height = 640, 480
fps = 30
frames_per_move = 30  # 1 second per command

# Movement command lists
ref_cmds = ["front", "left", "right", "reset"]
test_cmds = ["front", "left", "right", "reset","front", "left", "right", "reset","left", 
             "right", "front", "reset","front", "left", "right", "reset","left", "right", "front", "reset",
             "front", "left", "right", "reset","front", "left", "right", "reset","left", 
             "right", "front", "reset","front", "left", "right", "reset","left", "right", "front", "reset"]

# --- Rectangle settings ---
base_w, base_h = 100, 60
center_x, center_y = width // 2, height // 2

def generate_positions(commands):
    positions = []
    for cmd in commands:
        for i in range(frames_per_move):
            if cmd == "front":  # Move up
                dy = int(100 * (i / frames_per_move))
                positions.append((center_x, center_y - dy, base_w, base_h))
            elif cmd == "left":
                dx = int(100 * (i / frames_per_move))
                positions.append((center_x - dx, center_y, base_w, base_h))
            elif cmd == "right":
                dx = int(100 * (i / frames_per_move))
                positions.append((center_x + dx, center_y, base_w, base_h))
            elif cmd == "reset":
                positions.append((center_x, center_y, base_w, base_h))
            else:
                raise ValueError(f"Unknown command: {cmd}")
    return positions

# --- Generate positions ---
ref_positions = generate_positions(ref_cmds)
test_positions = generate_positions(test_cmds)

# --- Video Writers ---
fourcc = cv2.VideoWriter_fourcc(*'XVID')
ref_writer = cv2.VideoWriter(ref_path, fourcc, fps, (width, height))
test_writer = cv2.VideoWriter(test_path, fourcc, fps, (width, height))

def draw_rectangle(frame, cx, cy, w, h, color=(0, 255, 0)):
    top_left = (int(cx - w / 2), int(cy - h / 2))
    bottom_right = (int(cx + w / 2), int(cy + h / 2))
    cv2.rectangle(frame, top_left, bottom_right, color, -1)

# --- Write ref frames ONLY up to its own length ---
for i in range(len(ref_positions)):
    ref_frame = np.zeros((height, width, 3), dtype=np.uint8)
    rx, ry, rw, rh = ref_positions[i]
    draw_rectangle(ref_frame, rx, ry, rw, rh, color=(0, 255, 0))
    ref_writer.write(ref_frame)
ref_writer.release()

# --- Write test frames up to its own length ---
for i in range(len(test_positions)):
    test_frame = np.zeros((height, width, 3), dtype=np.uint8)
    tx, ty, tw, th = test_positions[i]
    draw_rectangle(test_frame, tx, ty, tw, th, color=(0, 255, 0))
    test_writer.write(test_frame)
test_writer.release()

print(f"✅ Ref video duration: {len(ref_positions)/fps:.2f} sec")
print(f"✅ Test video duration: {len(test_positions)/fps:.2f} sec")
print(f"Videos saved to:\n{ref_path}\n{test_path}")
