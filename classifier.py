import numpy as np
import cv2 as cv
import utils
from keras.models import load_model

class HandwritingClassifier:
    def __init__(self):
        self.MAP_SYMBOLS = {10: '+', 11: '-', 12: '*', 13: '/', 14: '(', 15: ')'}
        self.model = load_model('bestmodel.h5')

    def run(self, img):
        img = utils.prep_img(img)
        img = img.astype(np.float32) / 255.0 
        img = np.expand_dims(img, axis=-1)  
        result = self.model.predict(np.array([img]))[0]
        prediction = np.argmax(result)

        if prediction > 9:
            prediction = self.MAP_SYMBOLS[prediction]
        else:
            prediction = str(prediction)
        
        return prediction
