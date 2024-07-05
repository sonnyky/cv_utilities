import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Capture one frame
ret, frame = cap.read()

if not ret:
    print("Error: Could not read frame.")
else:
    # Display the captured image
    cv2.imshow('Captured Image', frame)

    # Save the captured image
    cv2.imwrite('images/captured_image.jpg', frame)

    # Wait for a key press and close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Release the webcam
cap.release()
