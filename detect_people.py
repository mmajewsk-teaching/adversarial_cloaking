import cv2
import numpy as np
import os
from PIL import Image
import apply_patch

CFG_PATH = "yolov2/yolo.cfg"
WEIGHTS_PATH = "yolov2/yolov2.weights"
IMAGE_PATH = "data/test_images"
OUTPUT_PATH = "data/patch_images"
PATCH_PATH = "output_patches/patch_final.png"
CONFIDENCE_THRESHOLD = 0.5                # Minimum confidence to accept a detection
NMS_THRESHOLD = 0.4                       # Non-Maximum Suppression threshold to remove overlapping boxes

"""Run YOLOv2 model on a specified image with displaying detection boxes and confidence"""
def detect_people(dir_path, filename):
    image_path = dir_path + "/" + filename
    if not os.path.exists(CFG_PATH) or not os.path.exists(WEIGHTS_PATH):
        print("Error: Could not find YOLOv2 configuration or weights files.")
        return

    print("Loading YOLOv2 model...")
    net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)

    try:
        pil_image = Image.open(image_path) 
        numpy_image = np.array(pil_image)
        
        # Convertion from RGB to BGR
        if len(numpy_image.shape) == 3 and numpy_image.shape[2] == 3:
            image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        else:
            # Grayscale or other format fallback
            image = cv2.cvtColor(numpy_image, cv2.COLOR_GRAY2BGR)
            
        print("Image loaded successfully with Pillow")
        
    except Exception as e:
        print(f"Pillow failed to open the image: {e}")
    
    (H, W) = image.shape[:2]

    # Scale image to 416x416 for YOLOv2
    # Normalize pixel values to [0, 1], swap Red and Blue
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Run forward pass
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    print("Running inference...")
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id == 0 and confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Convert center coordinates to top-left coordinates for OpenCV
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply NMS to eliminate duplicate overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw label and confidence
            text = f"Person: {confidences[i]:.2f}"
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            print(f"Detected Person - Probability: {confidences[i]:.2f}, Box: [X:{x}, Y:{y}, W:{w}, H:{h}]")

    else:
        print("No people detected above the confidence threshold.")

    cv2.imshow("YOLOv2 Person Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return boxes

"""Run YOLOv2 detection on images from specified directory"""
def run_detect_peole(dir_path):
    boxes_dict = {}
    for filename in os.listdir(dir_path):
        image_path = os.path.join(dir_path, filename)
        if os.path.isfile(image_path):
            boxes_dict[filename] = detect_people(dir_path, filename)
    return boxes_dict

if __name__ == "__main__":
    run_detect_peole("data/test_images")