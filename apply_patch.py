import cv2
import os
import numpy as np
import detect_people

"""Apply trained patches to detected objects on test images"""
def apply_patch(base_image_path, filename, patch_path, output_path, boxes):
    image_path = base_image_path +"/"+ filename
    base_data = np.fromfile(image_path, dtype=np.uint8)
    base_img = cv2.imdecode(base_data, cv2.IMREAD_COLOR)
    
    patch_data = np.fromfile(patch_path, dtype=np.uint8)
    patch_img = cv2.imdecode(patch_data, cv2.IMREAD_COLOR)

    if base_img is None:
        print(f"Error: Could not load base image at {base_image_path}")
        return
    if patch_img is None:
        print(f"Error: Could not load patch image at {patch_path}")
        return

    for curr_box in boxes:
        x, y, w, h = curr_box[0], curr_box[1], curr_box[2], curr_box[3]
        
        patch_w = max(1, int(w // 2))
        patch_h = max(1, int(h // 2))
        
        patch_resized = cv2.resize(patch_img, (patch_w, patch_h))

        patch_x = x + (w - patch_w) // 2
        patch_y = y + (h - patch_h) // 2

        img_h, img_w = base_img.shape[:2]
        
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

        base_img[start_y:end_y, start_x:end_x] = patch_resized[p_start_y:p_end_y, p_start_x:p_end_x]

    success, encoded_image = cv2.imencode('.jpg', base_img)
    if success:
        output_image_path = output_path +"/patch_"+ filename
        with open(output_image_path, 'wb') as f:
            encoded_image.tofile(f)
        print(f"Successfully saved patched image to: {output_image_path}")
    else:
        print("Error: Failed to save the image.")

def run_apply_patch(dir_path, patch_path, output_path, boxes_dict):
    for filename in os.listdir(dir_path):
        image_path = os.path.join(dir_path, filename)
        if os.path.isfile(image_path) and filename in boxes_dict:
            apply_patch(dir_path, filename, patch_path, output_path, boxes_dict[filename])

if __name__ == "__main__":
    boxes_dict = detect_people.run_detect_peole("data/test_images")
    run_apply_patch("data/test_images", "output_patches/patch_final.png", "data/patch_images", boxes_dict)
