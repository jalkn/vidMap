import cv2
import numpy as np

# 1. Initialize the webcam
cap = cv2.VideoCapture(0)

print("Mapping started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 2. Preprocessing
    # Convert to grayscale and blur slightly to reduce high-frequency noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # 4. Find Contours
    # We look for external shapes and simplify the lines
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort by area so we focus on the biggest objects first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        # Approximate the contour to a polygon
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 5. Check if the polygon has 4 corners (a rectangle/quadrilateral)
        if len(approx) == 4:
            # Draw the green outline on the original color frame
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            
            # Label the corners so we know the order for the next step
            for i, point in enumerate(approx):
                x, y = point[0]
                cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            break

    # 6. Display the result
    cv2.imshow('Surface Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()