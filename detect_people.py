import cv2
import numpy as np
import os

CFG_PATH = "yolov2/yolo.cfg"
WEIGHTS_PATH = "yolov2/yolov2.weights"

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

#model loading 
def load_model():
    print("Loading YOLOv2 model")

    net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)

    layer_names = net.getLayerNames()
    output_layers = [ layer_names[i - 1] for i in net.getUnconnectedOutLayers() ]

    return net, output_layers


def detect_people(net, output_layers, dir_path, filename):
    image_path = os.path.join(dir_path, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Skipping invalid image: {image_path}")
        return []

    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage( image, 1 / 255.0, (416, 416), swapRB=True, crop=False )

    net.setInput(blob)

    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # class 0 - person
            if class_id == 0 and confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        CONFIDENCE_THRESHOLD,
        NMS_THRESHOLD
    )

    final_boxes = []

    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])

    return (final_boxes, confidences)

def run_detect_peole(dir_path, max_images=None):
    boxes_dict = {}
    net, output_layers = load_model()

    files = [
        f for f in os.listdir(dir_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    files = sorted(files)

    if max_images is not None:
        files = files[:max_images]
    print(f"Running detection on {len(files)} images")

    for filename in files:
        boxes = detect_people(
            net,
            output_layers,
            dir_path,
            filename
        )
        boxes_dict[filename] = boxes
        print(f"{filename}: {len(boxes)} people detected")

    return boxes_dict

if __name__ == "__main__":
    results = run_detect_peole(
        "data/test_images",
        #max_images=5   #test 5 for now
    )
    print("\nDONE")