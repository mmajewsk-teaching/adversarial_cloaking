import cv2
import numpy as np
import os

CFG_PATH = "yolov2/yolo.cfg"
WEIGHTS_PATH = "yolov2/yolov2.weights"
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

def save_image(image, output_dir, filename):
    if image is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)

def load_image(input_dir, filename):
    input_path = os.path.join(input_dir, filename)
    image = cv2.imread(input_path)
    if image is None:
        print(f"Could not load image: {input_path}")
        return None
    return image

#model loading 
def load_model():
    print("Loading YOLOv2 model")
    net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
    layer_names = net.getLayerNames()
    output_layers = [ layer_names[i - 1] for i in net.getUnconnectedOutLayers() ]
    return net, output_layers

def apply_boxes(image, boxes, confidences):
    for (x, y, w, h), c in zip(boxes, confidences):
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
        cv2.putText(image, f"Person: {c:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)   
    return image

def detect_people(net, output_layers, image):
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

def run_detect_peole(input_dir, max_images=None):
    output_dir = input_dir + "_detected"
    boxes_dict = {}
    net, output_layers = load_model()

    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])

    if max_images is not None:
        files = files[:max_images]
    print(f"Running detection on {len(files)} images")

    for filename in files:
        image = load_image(input_dir, filename)
        boxes, confidences = detect_people(
            net,
            output_layers,
            image
        )
        image = apply_boxes(image, boxes, confidences)
        save_image(image, output_dir, "detected_" + filename)
        boxes_dict[filename] = boxes
        print(f"{filename}: {len(boxes)} people detected")

    print()
    return boxes_dict

if __name__ == "__main__":
    run_detect_peole("data/test_images")
    run_detect_peole("data/patch_images")

    print("DONE")