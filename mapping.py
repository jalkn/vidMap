import cv2

# 1. Initialize the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # 2. Convert to Grayscale
    # This reduces complexity by removing color information
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. Display the resulting frame
    cv2.imshow('Video Mapping - Step 1', gray)

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()