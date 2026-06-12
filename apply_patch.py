import cv2
import os
import numpy as np
import detect_people

"""Apply trained patches to detected objects on test images"""
def apply_patch(image, patch, boxes):
    if image is None:
        print(f"Error: Could not load base image")
        return
    if patch is None:
        print(f"Error: Could not load patch image")
        return

    for curr_box in boxes:
        x, y, w, h = curr_box[0], curr_box[1], curr_box[2], curr_box[3]
        
        patch_w = max(1, int(w // 2))
        patch_h = max(1, int(h // 2))
        
        patch_resized = cv2.resize(patch, (patch_w, patch_h))

        patch_x = x + (w - patch_w) // 2
        patch_y = y + (h - patch_h) // 2

        img_h, img_w = image.shape[:2]
        
        start_y = max(0, patch_y)
        start_x = max(0, patch_x)
        end_y = min(patch_y + patch_h, img_h)
        end_x = min(patch_x + patch_w, img_w)
        
        patch_crop_y = end_y - start_y
        patch_crop_x = end_x - start_x

        if patch_crop_y <= 0 or patch_crop_x <= 0:
            continue

        p_start_y = start_y - patch_y
        p_end_y = p_start_y + patch_crop_y
        p_start_x = start_x - patch_x
        p_end_x = p_start_x + patch_crop_x

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
    boxes_dict = detect_people.run_detect_people("data/test_images")
    run_apply_patch("data/test_images", "data/patch_images", "output_patches", "experimental.jpg", boxes_dict)
    print("DONE")
