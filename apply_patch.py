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


def apply_patch(base_image_path, filename, patch_path, output_path, boxes):
    image_path = os.path.join(base_image_path, filename)
    base_data  = np.fromfile(image_path, dtype=np.uint8)
    base_img   = cv2.imdecode(base_data, cv2.IMREAD_COLOR)

    patch_data = np.fromfile(patch_path, dtype=np.uint8)
    patch_img  = cv2.imdecode(patch_data, cv2.IMREAD_COLOR)

    if base_img is None:
        print(f"Error: Could not load base image at {base_image_path}")
        return
    if patch_img is None:
        print(f"Error: Could not load patch image at {patch_path}")
        return

    for curr_box in boxes[0]:
        x, y, w, h = curr_box[0], curr_box[1], curr_box[2], curr_box[3]

        patch_x, patch_y, patch_w, patch_h = get_patch_position(base_img, x, y, w, h)

        patch_resized = cv2.resize(patch_img, (patch_w, patch_h))

        img_h, img_w = base_img.shape[:2]
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

        base_img[start_y:end_y, start_x:end_x] = patch_resized[p_start_y:p_end_y, p_start_x:p_end_x]

    success, encoded_image = cv2.imencode('.jpg', base_img)
    if success:
        output_image_path = output_path + "/patch_" + filename
        with open(output_image_path, 'wb') as f:
            encoded_image.tofile(f)
        print(f"Saved: {output_image_path}")
    else:
        print("Error: Failed to save the image.")


def run_apply_patch(dir_path, patch_path, output_path, boxes_dict):
    files = [
        f for f in sorted(os.listdir(dir_path))
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        and os.path.isfile(os.path.join(dir_path, f))
    ]
    for filename in files:
        if filename in boxes_dict:
            print(f"\n{filename}")
            apply_patch(dir_path, filename, patch_path, output_path, boxes_dict[filename])


if __name__ == "__main__":
    os.makedirs("data/patch_images", exist_ok=True)
    boxes_dict = detect_people.run_detect_peole("data/test_images")
    run_apply_patch("data/test_images", "output_patches/patch_final.png", "data/patch_images", boxes_dict)