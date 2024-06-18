import cv2

# Open the default camera
cap = cv2.VideoCapture(0)

if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Save the captured image
        cv2.imwrite('captured_image.jpg', frame)

    cap.release()
else:
    print("Error: Could not open camera")
