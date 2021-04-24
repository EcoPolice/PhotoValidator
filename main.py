from flask import Flask, request, jsonify

app = Flask(__name__)

from tensorflow import keras

model = keras.models.load_model('model')

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array

import cv2


def predict_test(filename):
    data = []
    img_read = plt.imread(filename)
    img_resize = cv2.resize(img_read, (100, 100))
    img_array = img_to_array(img_resize)
    img_array = img_array / 255
    data.append(img_array)
    image_data = np.array(data)
    idx = np.arange(image_data.shape[0])
    np.random.shuffle(idx)
    image_data = image_data[idx]
    return model.predict(image_data).argmax(axis=1)[0]


@app.route('/', methods=['POST'])
def index():
    request_data = request.get_json(force=True)
    print(request_data)
    try:
        return jsonify({'error': False, 'is_oil': int(predict_test(request_data['filepath'])) }), 200
    except Exception as e:
        return jsonify({'error': True, 'message': str(e)}), 200
