from flask import Flask, request, jsonify
from tensorflow import keras
import numpy as np
from keras.preprocessing.image import img_to_array
import imghdr
import cv2

app = Flask(__name__)
model = keras.models.load_model('model')


def check_photo(file_name):
    if imghdr.what(file_name) is None:
        raise Exception("Bad photo")


def predict_test(filename):
    data = []
    check_photo(filename)
    try:
        img_read = cv2.imread(filename)
    except cv2.error as e:
        raise Exception('Bad photo')
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
        return jsonify({'error': False, 'is_oil': int(predict_test(request_data['filepath']))}), 200
    except Exception as e:
        return jsonify({'error': True, 'message': str(e)}), 200
