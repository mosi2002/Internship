import cv2
import os
import numpy as np
import tensorflow as tf

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Set the directory to save the person's images
person_name = 'mostafa'
person_dir = r'C:\Users\Ashkan\Desktop\MOSTAFA'

if not os.path.exists(person_dir):
    os.makedirs(person_dir)


# Set the number of images to capture
num_images = 30

# Create a list to store the captured face images
face_images = []

region_x = 0
region_y = 0

# Initial circle region
circle_region_x = 0
circle_region_y = 0

# Create a video capture object for the camera
cap = cv2.VideoCapture(0)

# Capture and process frames from the video stream
capture_count = 0
while True:
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    height, width, _ = frame.shape
    region_width = width // 3
    region_height = height // 3
    region_x = circle_region_x * region_width
    region_y = circle_region_y * region_height
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Crop the face region from the frame
        face = gray[y:y+h, x:x+w]
        
        # Resize the face image to a fixed size (e.g., 64x64)
        face = cv2.resize(face, (64, 64))

        for i in range(3):
            for j in range(3):
                x = j * region_width
                y = i * region_height
                center_x = x + region_width // 2
                center_y = y + region_height // 2

                if i == circle_region_y and j == circle_region_x:
                    cv2.circle(frame, (center_x, center_y), 50, (0, 0, 255), 2)
        
        # Display the captured face image
        cv2.imshow('Captured Face', face)
        
        # Check if the 'y' key is pressed to capture the face image
        if cv2.waitKey(1) & 0xFF == ord('y'):
            circle_region_x += 1
            if circle_region_x >= 3:
                circle_region_x = 0
                circle_region_y += 1
                if circle_region_y >= 3:
                    circle_region_y = 0
            # Add the face image to the list
            face_images.append(face)
            
            # Save the face image to the person's directory
            cv2.imwrite(os.path.join(person_dir, f'{person_name}_{capture_count}.jpg'), face)
            
            # Increment the capture count
            capture_count += 1
            
            # Break the loop if the desired number of images is captured
            if capture_count >= num_images:
                break
    
    # Display the processed frame
    cv2.imshow('Video', frame)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()