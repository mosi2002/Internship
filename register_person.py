import cv2
from deepface import DeepFace
import os
from update_pickle import update_pickl
import id_generator

# Set the directory to save the person's images
path = r"C:\Users\Ashkan\Desktop\Face_recognition2\Face_recognition.pkl"
person_dir = r'C:\Users\Ashkan\Desktop\Face_recognition2\Dataset'

if not os.path.exists(person_dir):
    os.makedirs(person_dir)
# Load the face recognition model from DeepFace
    
def draw_rectangle_extract_ROI(frame):
    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # Define the rectangle's parameters for the middle area
    rect_top = height // 4
    rect_bottom = 3 * height // 4
    rect_left = width // 5 + 100
    rect_right = 3 * width // 5 + 100

    # Draw the rectangle on the frame
    cv2.rectangle(frame, (rect_left, rect_top), (rect_right, rect_bottom), (0, 255, 0), 2)

    # Extract the region of interest (ROI) within the rectangle
    roi = frame[rect_top:rect_bottom, rect_left:rect_right]
    return roi


camera = cv2.VideoCapture(0)  # 0 represents the default camera
text = "Press 'y' to capture"
while True:
    # Capture a frame from the camera
    ret, frame = camera.read()
    if not ret:
        break
    roi = draw_rectangle_extract_ROI(frame)

    # Check if the 'y' key is pressed to capture the face image
    if cv2.waitKey(1) & 0xFF == ord('y'):
        try:
            print("captured")
            # Save the face image to the person's directory
            new_rep_one = DeepFace.represent(img_path=roi, model_name="VGG-Face", detector_backend="opencv")
            new_rep_two = DeepFace.represent(img_path=roi, model_name="Facenet", detector_backend="opencv")
            vectors = [new_rep_one, new_rep_two]
            name = input("person_name: ")
            cv2.imwrite(os.path.join(person_dir, f'{name}.jpg'), roi)
            fingerprint_hex = "h12345"
            personal_code = "39213501231"
            person_id = id_generator.uuid_id_generator()
            update_pickle(person_id, name, personal_code, vectors, fingerprint_hex, path)
        except:
            print("can't find face, capture again")
    cv2.imshow('Video', frame)

# Release the camera and close the windows
camera.release()
cv2.destroyAllWindows()

