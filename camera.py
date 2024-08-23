import cv2
from RecognitnionPhase import perform_recognition_phase_one_two
from deepface import DeepFace
import pickle
from update_pickle import read_pickle
import numpy as np

path = "/home/ftm/Downloads/Face_recognition2/Face_recognition.pkl"
first_threshold = 0.2
second_threshold = 0.1
model_one = "VGG-Face"
model_two = "Facenet"

def draw_rectangle_extract_ROI(frame):

    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # Define the rectangle's parameters for the middle area
    rect_top = height // 4
    rect_bottom = 3 * height // 4
    rect_left = width // 5 + 100
    rect_right = 3 * width // 5 + 100

    # Draw the rectangle on the frame
    # cv2.rectangle(frame, (rect_left, rect_top), (rect_right, rect_bottom), color, 2)

    # Extract the region of interest (ROI) within the rectangle
    roi = frame[rect_top:rect_bottom, rect_left:rect_right]
    return roi, rect_left, rect_top, rect_right, rect_bottom


def read_data(directory_path):
    with open(directory_path, 'rb') as f:
        df = pickle.load(f)
    return df


def extract_face_and_representaion(frame, model_name):
    target_size = DeepFace.function.find_target_size(model_name = model_name)
    img_objs = DeepFace.functions.extract_faces(
        img=frame,
        target_size=target_size,
        detector_backend="opencv",
        grayscale=False,
        enforce_detection=True,
        align=True)


def run_camera():
    data_frame = read_pickle(path)
    print(data_frame)
    camera = cv2.VideoCapture(0)  # 0 represents the default camera
    text = ""  # the bounding box text
    while True:
        # Capture a frame from the camera
        ret, frame = camera.read()
        #check if camera exits
        if not ret:
            break
        # print(np.array(frame.shape))
        roi, rect_left, rect_top, rect_right, rect_bottom = draw_rectangle_extract_ROI(frame)
        # print(roi)
        # img = DeepFace.functions.FaceDetector.detect_faces(frame)

        try:
            # elected_df = perform_recognition_phase_one_two(data_frame, first_threshold, model_one, roi)
            name = perform_recognition_phase_one_two(data_frame, first_threshold, model_one,roi)
            if name == 2:
                name = perform_recognition_phase_one_two(data_frame, second_threshold, model_two, roi)
            text = name
            if name == "unknown" :
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
        except:
            text = "No face"
            color = (0, 140, 255)
       # check recognized_person = recognize_face(roi)
        print(text)
        cv2.putText(frame, text, (rect_left, rect_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (rect_left, rect_top), (rect_right, rect_bottom), color, 2)

        cv2.imshow('Frame', frame) # shows the frame

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Release the camera and close the windows
    camera.release()
    cv2.destroyAllWindows()


run_camera()
