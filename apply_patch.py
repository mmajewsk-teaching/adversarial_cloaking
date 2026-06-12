import cv2
import os
import numpy as np
import detect_people

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def get_patch_position(base_img, x, y, w, h):
    img_h, img_w = base_img.shape[:2]

    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    if w <= 0 or h <= 0:
        chest_start = int(h * 0.22)
        chest_end   = int(h * 0.62)
        body_h      = chest_end - chest_start
        patch_w     = max(1, int(w * 0.55))
        patch_h     = max(1, int(body_h * 0.80))
        patch_x     = x + (w - patch_w) // 2
        patch_y     = y + chest_start
        return patch_x, patch_y, patch_w, patch_h

    roi = base_img[y:y+h, x:x+w]

    if roi.size == 0:
        chest_start = int(h * 0.22)
        chest_end   = int(h * 0.62)
        body_h      = chest_end - chest_start
        patch_w     = max(1, int(w * 0.55))
        patch_h     = max(1, int(body_h * 0.80))
        patch_x     = x + (w - patch_w) // 2
        patch_y     = y + chest_start
        return patch_x, patch_y, patch_w, patch_h

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(20, 20)
    )

    if len(faces) > 0:
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_bottom = fy + fh
        margin      = int(fh * 0.2)
        patch_h     = max(1, int(h * 0.35))
        patch_w     = max(1, int(w * 0.55))
        patch_y     = y + face_bottom + margin
        patch_y     = min(patch_y, y + int(h * 0.55))
        patch_x     = x + (w - patch_w) // 2
        print(f"  twarz wykryta — patch pod twarzą (bbox_y={y}, face_bottom={y+face_bottom})")
    else:
        chest_start = int(h * 0.12)
        chest_end   = int(h * 0.50)
        body_y      = y + chest_start
        body_h      = chest_end - chest_start
        patch_w     = max(1, int(w * 0.55))
        patch_h     = max(1, int(body_h * 0.80))
        patch_x     = x + (w - patch_w) // 2
        patch_y     = body_y
        print(f"  brak twarzy — fallback procentowy (chest {chest_start}–{chest_end}px od góry bbox)")

    return patch_x, patch_y, patch_w, patch_h


def apply_patch(image, patch, boxes):
    if image is None:
        print(f"Error: Could not load base image")
        return
    if patch is None:
        print(f"Error: Could not load patch image")
        return

    for curr_box in boxes:
        x, y, w, h = curr_box[0], curr_box[1], curr_box[2], curr_box[3]

        patch_x, patch_y, patch_w, patch_h = get_patch_position(image, x, y, w, h)

        patch_resized = cv2.resize(patch, (patch_w, patch_h))

        img_h, img_w = image.shape[:2]
        start_y = max(0, patch_y)
        start_x = max(0, patch_x)
        end_y   = min(patch_y + patch_h, img_h)
        end_x   = min(patch_x + patch_w, img_w)

        patch_crop_y = end_y - start_y
        patch_crop_x = end_x - start_x

        if patch_crop_y <= 0 or patch_crop_x <= 0:
            continue

        p_start_y = start_y - patch_y
        p_end_y   = p_start_y + patch_crop_y
        p_start_x = start_x - patch_x
        p_end_x   = p_start_x + patch_crop_x

        image[start_y:end_y, start_x:end_x] = patch_resized[p_start_y:p_end_y, p_start_x:p_end_x]

    return image


def run_apply_patch(input_dir, output_dir, patch_dir, patch_name, boxes_dict):
    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])

    for filename in files:
        boxes = boxes_dict[filename]
        image = detect_people.load_image(input_dir, filename)
        patch = detect_people.load_image(patch_dir, patch_name)
        image = apply_patch(image, patch, boxes)
        
        detect_people.save_image(image, output_dir, "patch_" + filename)

if __name__ == "__main__":
    os.makedirs("data/patch_images", exist_ok=True)
    boxes_dict = detect_people.run_detect_people("data/test_images")
    run_apply_patch("data/test_images", "data/patch_images", "output_patches", "experimental.jpg", boxes_dict)
    print("DONE")
