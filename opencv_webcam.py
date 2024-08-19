# for webcam, kinda good quality
import cv2
import numpy as np

# Define the range for the tennis ball color in HSV
lower_yellow_green = np.array([25, 70, 120])
upper_yellow_green = np.array([60, 255, 255])

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Convert the frame to the HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for the tennis ball color
        mask = cv2.inRange(hsv, lower_yellow_green, upper_yellow_green)

        # Perform morphological operations to remove noise and improve detection
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Track the tennis ball by finding the largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area > 1000:  # Adjust the area threshold as needed
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                center = (int(x), int(y))
                radius = int(radius)

                # Draw the circle and centroid on the frame
                cv2.circle(frame, center, radius, (0, 255, 0), 4)
                cv2.circle(frame, center, 5, (0, 128, 255), -1)

        # Display the resulting frame
        cv2.imshow("Tennis Ball Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
