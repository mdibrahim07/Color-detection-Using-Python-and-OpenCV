import cv2
import numpy as np
from PIL import Image

# Define the HSV color ranges for each color
def get_limits(color):
    if color == [0, 255, 255]:  # yellow in BGR
        lower = np.array([20, 100, 100], dtype=np.uint8)
        upper = np.array([30, 255, 255], dtype=np.uint8)
    elif color == [255, 0, 0]:  # blue in BGR
        lower = np.array([100, 150, 0], dtype=np.uint8)
        upper = np.array([140, 255, 255], dtype=np.uint8)
    elif color == [0, 255, 0]:  # green in BGR
        lower = np.array([40, 40, 40], dtype=np.uint8)
        upper = np.array([70, 255, 255], dtype=np.uint8)
    else:
        lower = upper = None
    return lower, upper

# Define colors in BGR color space
colors = {
    'yellow': [0, 255, 255],
    'blue': [255, 0, 0],
    'green': [0, 255, 0]
}

# Open a connection to the default camera (index 0)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame from the camera
    ret, frame = cap.read()
    
    # Check if frame was captured successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the captured frame from BGR to HSV color space
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Iterate over defined colors
    for color_name, bgr_value in colors.items():
        # Get the lower and upper HSV limits for the current color
        lowerLimit, upperLimit = get_limits(color=bgr_value)
        
        # Create a mask for the current color
        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

        # Convert the mask to a PIL Image object to use getbbox()
        mask_ = Image.fromarray(mask)

        # Get the bounding box coordinates of the non-zero regions in the mask
        bbox = mask_.getbbox()

        # If a bounding box is found, draw a rectangle on the original frame
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            rect_color = tuple(bgr_value)  # Set the rectangle color to match the detected color
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 5)
            # Add the color name label near the top-left corner of the rectangle
            cv2.putText(frame, color_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rect_color, 2)

    # Display the frame with the rectangles and labels (if drawn)
    cv2.imshow('frame', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from PIL import Image

# # Define the HSV color ranges for each color
# def get_limits(color):
#     if color == [0, 255, 255]:  # yellow in BGR
#         lower = np.array([25, 150, 150], dtype=np.uint8)
#         upper = np.array([35, 255, 255], dtype=np.uint8)
#     elif color == [255, 0, 0]:  # blue in BGR
#         lower = np.array([90, 100, 100], dtype=np.uint8)
#         upper = np.array([130, 255, 255], dtype=np.uint8)
#     elif color == [0, 255, 0]:  # green in BGR
#         lower = np.array([50, 100, 100], dtype=np.uint8)
#         upper = np.array([80, 255, 255], dtype=np.uint8)
#     else:
#         lower = upper = None
#     return lower, upper

# # Define colors in BGR color space
# colors = {
#     'yellow': [0, 255, 255],
#     'blue': [255, 0, 0],
#     'green': [0, 255, 0]
# }

# # Open a connection to the default camera (index 0)
# cap = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame from the camera
#     ret, frame = cap.read()
    
#     # Check if frame was captured successfully
#     if not ret:
#         print("Error: Could not read frame.")
#         break

#     # Convert the captured frame from BGR to HSV color space
#     hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Apply Gaussian blur to reduce noise
#     hsvImage = cv2.GaussianBlur(hsvImage, (5, 5), 0)

#     # Iterate over defined colors
#     for color_name, bgr_value in colors.items():
#         # Get the lower and upper HSV limits for the current color
#         lowerLimit, upperLimit = get_limits(color=bgr_value)
        
#         # Create a mask for the current color
#         mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

#         # Convert the mask to a PIL Image object to use getbbox()
#         mask_ = Image.fromarray(mask)

#         # Get the bounding box coordinates of the non-zero regions in the mask
#         bbox = mask_.getbbox()

#         # If a bounding box is found, draw a rectangle on the original frame
#         if bbox is not None:
#             x1, y1, x2, y2 = bbox
#             rect_color = tuple(bgr_value)  # Set the rectangle color to match the detected color
#             frame = cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 5)
#             # Add the color name label near the top-left corner of the rectangle
#             cv2.putText(frame, color_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rect_color, 2)

#     # Display the frame with the rectangles and labels (if drawn)
#     cv2.imshow('frame', frame)

#     # Break the loop if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()