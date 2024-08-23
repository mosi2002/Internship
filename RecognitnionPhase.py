from deepface import DeepFace
import cv2
from deepface.commons import functions, realtime, distance as dst
import pickle
import pandas as pd
import json
import os

first_threshold = 0.4
second_threshold = 0.6
directory_path = ""


# def read_data():
#     with open(directory_path, 'rb') as f:
#         df = pickle.load(f)
#     return df

# def save_data():
#     with open(directory_path, 'wb') as f:
#         df = pickle.dumb(f)
#     return df
    
# phase one and two recognition
def perform_recognition_phase_one_two(data_frame, threshold, model, img_objs):
    elected_df = pd.DataFrame(columns=data_frame.columns)
    # read all the representations and measure the distances
    # for img_content, _, _ in img_objs:
    new_rep = DeepFace.represent(img_path=img_objs, model_name=model, detector_backend="opencv")
    # print(new_rep)
        # person_id, name, personal_code, vectors, fingerprint_hex
    for id , name, personal_code, vectors, _ in zip(data_frame["person_id"], data_frame["name"], data_frame["personal_code"], data_frame["vectors"], data_frame["fingerprint_hex"]):
        is_close = compare_representations(new_rep, vectors[0], threshold)
        if is_close is True:
            elected_df.loc[len(elected_df)] = [id, name, personal_code, vectors[0], None]

    if len(elected_df) > 1:
        return 2
    elif len(elected_df) == 1:
        return elected_df['name'].iloc[0]
    else:
        return "unknown"
    
# calculate distance and compare the representations for real time compareson
def compare_representations(new_representaion, target_representaion, threshold):
    # print(target_representaion.shape)
    # print(new_representaion.shape)
    # print(target_representaion)
    distance = dst.findCosineDistance(new_representaion[0]["embedding"], target_representaion[0]["embedding"])
    print("distance: ", distance)
    return (True if distance < threshold else False)


# phase two recognition
def perform_recognition_phase_two(elected_df, new_rep):
    df = pd.DataFrame()
    face_average_measure = []
    for id, name, personal_code, target_vector in zip(elected_df["id"], elected_df["name"], elected_df["personal_code"], elected_df["vector"]):
        is_close = compare_representations(new_rep, target_vector[0], second_threshold)
        if is_close is True:
            row = zip(elected_df.columns,[id, name, personal_code, target_vector[0]])
            df = elected_df.append(row, ignore_index = True)
    return (df if len(elected_df) > 0 else None)

    
# exract representations and save the first profile image
def extract_representation(images, profile_folder):
    cv2.imwrite(profile_folder, images)
    representations = []
    representations.append(images.drop(0))
    representations.append(DeepFace.represent(img_path=image, model_name="VGG-Face", detector_backend="opencv"))
    for image in images:
        representations.append(DeepFace.represent(img_path=image, model_name="Facenet", detector_backend="opencv"))
    return representations


