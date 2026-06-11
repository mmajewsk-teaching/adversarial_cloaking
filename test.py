import detect_people
import cv2
import os

DIR = "data/patch_images"

def show_results(dir_path, max_images=5):
    net, output_layers = detect_people.load_model()
    
    files = [
        f for f in os.listdir(dir_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    files = sorted(files)[:max_images]

    for filename in files:
        image_path = os.path.join(dir_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue
        boxes = detect_people.detect_people(
            net,
            output_layers,
            dir_path,
            filename
        )
        # draw boxes
        for (x, y, w, h) in boxes:
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )
        cv2.imshow(f"Detection: {filename}", image)

        print(f"{filename}: {len(boxes)} people")
        cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_results(DIR, max_images=5)