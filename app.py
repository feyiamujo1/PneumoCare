from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import keras
from keras.models import load_model
import tensorflow_hub as hub
import numpy as np

model = load_model("VGG16_pneumonia_model.h5", custom_objects={'KerasLayer': hub.KerasLayer})


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/'

@app.route('/', methods=['GET'])
def home():
    # Get the URL of the logo image using url_for
    logo_url = url_for('static', filename='images/logo.png')
    main_image_url = url_for('static', filename='images/hero.jpg')
    return render_template('index.html', logo_url=logo_url, main_image_url=main_image_url)

@app.route('/', methods=['POST'])
def predict():


    # Get Image from frontend
    imageFile = request.files['imageFile'] 
    imageFileName = "input_image.jpg"
    image_path = 'static/images/'+ imageFileName
    imageFile.save(image_path)

    # model_path = 'VGG16_pneumonia_model.h5'
    # model = keras.models.load_model(model_path)

    #Preprocess input image
    uploaded_image = url_for('static', filename='images/input_image.jpg')
    input_image = cv2.imread("static/images/input_image.jpg")
    input_image_resized = cv2.resize(input_image, (224, 224))
    scaled_input_image = input_image_resized/255
    reshaped_input_image = np.reshape(scaled_input_image, [1, 224, 224, 3])

    # Model prediction here
    model_prediction = model.predict(reshaped_input_image)
    prediction_label = (model_prediction> 0.5).astype("int32").flatten()

    if prediction_label == 0:
        result = "Pneumonia"
    else:
        result = "Normal"

    # Get the URL of the logo image using url_for
    logo_url = url_for('static', filename='images/logo.png') 
    main_image_url = url_for('static', filename='images/hero.jpg')

    return render_template('index.html', logo_url=logo_url, main_image_url=main_image_url, loaded_image=uploaded_image, result=result)


if __name__ == '__main__':
    app.run(debug=True)