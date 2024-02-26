import cv2
import numpy as np

# Function to create a region of interest mask
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

# Function to draw lines on the image
def draw_lines(img, lines, color=(0, 0, 255), thickness=2):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Function to draw middle lines between detected contours
def draw_middle_lines(img, contours, color=(0, 255, 0), thickness=3):
    if contours is not None and len(contours) > 1:
        for i in range(len(contours) - 1):
            M1 = cv2.moments(contours[i])
            M2 = cv2.moments(contours[i + 1])

            if M1["m00"] != 0 and M2["m00"] != 0:
                cx1, cy1 = int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"])
                cx2, cy2 = int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"])

                # Calculate midpoint
                mid_x = (cx1 + cx2) // 2
                mid_y = (cy1 + cy2) // 2

                # Draw line
                cv2.line(img, (cx1, cy1), (cx2, cy2), color, thickness)

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

# Main loop to process each frame from the camera
while True:
    # Capture the frame
    ret, frame = cap.read()
    
    # Break the loop if there's no frame
    if not ret:
        break

    # Define the region of interest as a slightly larger square
    height, width = frame.shape[:2]
    roi_size = 70  # Increase the size of the ROI
    roi_vertices = np.array([[(width // 2 - roi_size, height // 2 - roi_size),
                              (width // 2 + roi_size, height // 2 - roi_size),
                              (width // 2 + roi_size, height // 2 + roi_size),
                              (width // 2 - roi_size, height // 2 + roi_size)]], dtype=np.int32)

    # Draw the region of interest on the frame
    cv2.polylines(frame, roi_vertices, isClosed=True, color=(0, 255, 0), thickness=2)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to the grayscale image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Apply region of interest mask
    roi = region_of_interest(edges, roi_vertices)

    # Find contours in the region of interest
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original frame
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

    # Draw middle lines on the frame
    draw_middle_lines(frame, contours)

    # Display the result
    cv2.imshow('Middle Lines Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
