import pickle
import pandas as pd
import os


path = r"C:\Users\Ashkan\Desktop\Face_recognition2\Face_recognition.pkl"

def read_pickle(filename):
    try:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        with open(filename, 'wb') as empty_pickle_file:
            empty_data = {}
            pickle.dump(empty_data, empty_pickle_file)  
            return None     
    

def write_pickle(data, filename):
    try:
        with open(filename , 'wb') as file:
            pickle.dump(data, file)
            print(f"Data written to '{filename}' successfully.")
    except Exception as e:
        print("An error occurred:", e)


# data  = { "id": "p1000", "name": "mostafa", "personal_code": "123456", "vector": "", "finger_print": "" }

def update_pickle(person_id, name, personal_code, vectors, fingerprint_hex, filename):
    old_df = read_pickle(filename)
    new_df = old_df.copy()
    new_df.loc[len(new_df)] = [person_id, name, personal_code, vectors, fingerprint_hex]

    write_pickle(new_df, filename)



if __name__ == "__main__":
    # Column names
    columns = ["person_id", "name", "personal_code", "vectors", "fingerprint_hex"]
    print(read_pickle(path))
















