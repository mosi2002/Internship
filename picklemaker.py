import cv2
import os
import bz2
import pickle
import numpy as np
import sklearn
#from keras.utils import np_utils

dir = r"C:\Users\Ashkan\Desktop\DetectingOperator"
categ = ['messi', 'ronaldo']

def create_data():
    image_list = []
    label_list = []
    for category in categ:
        path = os.path.join(dir, category)
        class_name = categ.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                image_list.append(img_array)
                label_list.append(class_name)

            except Exception as e:
                pass
    return np.array(image_list), np.array(label_list)

x_train , y_train = create_data()

x_train , y_train = sklearn.utils.shuffle(x_train , y_train)
x_train = x_train/255.0
#y_train = np_utils.to_categorical(y_train, 2)
dataset = [x_train, y_train]
print(x_train.shape)
with bz2.BZ2File("ali_data.pkl", "wb") as f:
    pickle.dump(dataset, f)

# np.save(r"C:\Users\Ashkan\Desktop\y_train", y_train)