import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    # 1. Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 2. Preprocessing: Convert to Grayscale
    # Simplifies the image for the edge detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. Edge Detection: Canny
    # This identifies the outlines of objects
    # 50 = low threshold, 150 = high threshold
    edges = cv2.Canny(gray, 50, 150)

    # 4. Display the results
    # You can change 'edges' back to 'frame' to see the original video
    cv2.imshow('Video Mapping - Edge Detection', edges)

    # Exit logic
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()