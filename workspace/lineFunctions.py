import cv2
import numpy as np

# Function to create a region of interest mask
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

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
