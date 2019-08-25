from copy import deepcopy

import cv2
import numpy as np
from keras.engine.saving import load_model

import utils

WEIGHT_PATH = 'models/checkpoints/weights_051.hdf5'
BASE_TEST_DATA = 'test/'

def predict():
    model = load_model(WEIGHT_PATH)
    
    while True:
        path_to_image = str(input('Image name: '))
        path_to_image = BASE_TEST_DATA+path_to_image
        
        img = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
        orig_img = deepcopy(img)
        img = cv2.resize(img, (68, 68))
        img = np.multiply(img, 1/255)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
    
    
        prediction = model.predict(img)
    
        index_of_prediction = np.argmax(prediction, axis=1)[0]
        emotion = utils.classes_to_emotions_dict[index_of_prediction]
    
        cv2.imshow(emotion, orig_img)
        cv2.waitKey(0)

if __name__ == '__main__':
    predict()