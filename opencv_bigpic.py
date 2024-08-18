# for downscaling pics when they are too big
import cv2
import numpy as np

# Load the image
image = cv2.imread('tennis_ball3.jpg')

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image")
else:
    # Resize if necessary
    max_dimension = 1000
    if image.shape[0] > max_dimension or image.shape[1] > max_dimension:
        scale_factor = max_dimension / float(max(image.shape[0], image.shape[1]))
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    output = image.copy()

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Perform histogram equalization on the value channel to reduce shadows
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

    # Define the range for the tennis ball color
    lower_yellow_green = np.array([25, 70, 120])
    upper_yellow_green = np.array([60, 255, 255])

    # Create a mask for the tennis ball color
    mask = cv2.inRange(hsv, lower_yellow_green, upper_yellow_green)

    # Show the mask for debugging
    cv2.imshow("Mask", mask)
    
    # Perform morphological operations to remove small shadows
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Show the cleaned mask for debugging
    cv2.imshow("Cleaned Mask", mask)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(mask, 50, 150)

    # Show the edges for debugging
    cv2.imshow("Edges", edges)

    # Combine the color mask and edge detection
    combined = cv2.bitwise_and(mask, edges)

    # Show the combined result for debugging
    cv2.imshow("Combined Mask and Edges", combined)

    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size to focus on the tennis ball
    for contour in contours:
        area = cv2.contourArea(contour)
        print(f"Contour Area: {area}")  # Debugging: print contour area
        if 1000 < area < 10000:  # Adjust area thresholds as needed
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(output, center, radius, (0, 255, 0), 4)

    # Show the final output
    cv2.imshow("Tennis Ball Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
