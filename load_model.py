import numpy as np
from keras.models import model_from_json
import cv2


def test(img: np.ndarray):
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    print(img.shape)
    reshaped_img = cv2.resize(img, (224,224))
    print(np.array([reshaped_img]).shape)
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model.predict(np.array([reshaped_img]))