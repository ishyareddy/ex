import cv2
import json
import os

annotations = []
drawing = False
ix, iy = -1, -1
ref_img = None
img_copy = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, annotations, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = ref_img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow(window_name, img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        label = input(f"Enter label for box ({x1}, {y1}) to ({x2}, {y2}): ")
        annotations.append({
            "label": label,
            "box": [x1, y1, x2, y2]
        })
        cv2.rectangle(ref_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img_copy = ref_img.copy()
        cv2.imshow(window_name, img_copy)

def annotate_reference_image(image_path, save_path):
    global ref_img, img_copy, window_name

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    ref_img = cv2.imread(image_path)
    if ref_img is None:
        print("❌ Failed to read image")
        return

    img_copy = ref_img.copy()
    window_name = "🖼️ Draw bounding boxes - press 's' to save, 'q' to quit"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_rectangle)

    print("🔧 Instructions:")
    print("- Click and drag to draw a box")
    print("- Label it in the terminal")
    print("- Press 's' to save, or 'q' to quit")

    while True:
        cv2.imshow(window_name, img_copy)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            data = {a['label']: {'box': a['box']} for a in annotations}
            with open(save_path, "w") as f:
                json.dump(data, f, indent=4)
            print(f"✅ Saved to {save_path}")
            break
        elif key == ord('q'):
            print("👋 Exiting without saving")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    annotate_reference_image("data/ref_img.png", "config/layout.json")
