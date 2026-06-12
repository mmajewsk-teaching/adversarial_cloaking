import detect_people
import cv2
import os

DIR = "data/patch_images"

def show_results(dir_path, max_images=None):
    output_path = dir_path + "_detected"
    print(f"Input: {dir_path}, Output: {output_path}")
    net, output_layers = detect_people.load_model()
    
    files = [
        f for f in os.listdir(dir_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    files = sorted(files) if max_images is None else sorted(files)[:max_images]

    for filename in files:
        image_path = os.path.join(dir_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue
        (boxes, confidences) = detect_people.detect_people(
            net,
            output_layers,
            dir_path,
            filename
        )
        # draw boxes
        for (x, y, w, h), c in zip(boxes, confidences):
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )
            cv2.putText(image, f"Person: {c:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"{filename}: {len(boxes)} people")

        success, encoded_image = cv2.imencode('.jpg', image)

        if not success:
            print(f"Failed to process image: {filename}")
            return
        
        name = "detected_" + filename
        output_image_path = os.path.join(output_path, name)
        os.makedirs(output_path, exist_ok=True)
        with open(output_image_path, 'wb') as f:
            encoded_image.tofile(f)

if __name__ == "__main__":
    show_results("data/test_images")
    show_results("data/patch_images")