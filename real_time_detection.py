import cv2
import detect_people

def fun(camera_device=0):
    window_name = "Real-Time Detection"
    cap = cv2.VideoCapture(camera_device)
    if not cap.isOpened():
        print("Error: Could not connect to the webcam.")
        return
    print("Camera connected")
    print("===== Press q to exit =====")

    net, output_layers = detect_people.load_model()
    

    while True:
        ret, frame = cap.read() 
        if not ret:
            break
        
        image = cv2.flip(frame, 1)
        boxes, confidences = detect_people.detect_people(net, output_layers, image)
        image = detect_people.apply_boxes(image, boxes, confidences)

        cv2.imshow(window_name, image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    # 5. Clean up safely when the loop breaks
    cap.release()
    cv2.destroyAllWindows()
    print("Window closed by user")

if __name__ == "__main__":
    fun()