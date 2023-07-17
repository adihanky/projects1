# # from flask import Flask, render_template
# # import tensorflow as tf
# # from tensorflow import keras
# # from tensorflow.keras.models import Sequential, load_model
# # from keras.models import load_model
# # from PIL import Image
# # import numpy as np
# # from executor.unet_inferrer import UnetInferrer
# # import cv2
# # app = Flask(__name__)
# # def prepare(filepath):
# #     IMG_SIZE = 224
# #     img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
# #     new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
# #     return new_array.reshape(-3,IMG_SIZE,IMG_SIZE,3)

# # def get_model():
# #     global model
# #     model = load_model('model_vgg16.h5')
# #     print(" * Model loaded!")

# # @app.route('/')
# # def index():
# #     return render_template("index.html")
    


# # @app.route('/predict',methods=["POST","GET"])
# # def predict():
# #     get_model()
    
# #     return "val"


# # if __name__ == '__main__':
# #     app.run(host="localhost", port=8080)

# #####################################################################################
# #app.py
# from flask import Flask, flash, request, redirect, url_for, render_template
# import urllib.request
# import os
# from werkzeug.utils import secure_filename
 
# app = Flask(__name__)
 
# UPLOAD_FOLDER = 'static/uploads/'
 
# app.secret_key = "secret key"
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
# @app.route('/')
# def home():
#     return render_template('index.html')
 
# @app.route('/', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         flash('No file part')
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         flash('No image selected for uploading')
#         return redirect(request.url)
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         #print('upload_image filename: ' + filename)
#         flash('Image successfully uploaded and displayed below')
#         return render_template('index.html', filename=filename)
#     else:
#         flash('Allowed image types are - png, jpg, jpeg, gif')
#         return redirect(request.url)


# if __name__ == "__main__":
#     app.run()        

from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
CATEGORIES = ["Normal","Pneumonia"]
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model_vgg16.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
# print('Model loaded. Start serving...')



#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        if preds[0][0] > 0. :
            result = CATEGORIES[0]
        else:
            result = CATEGORIES[1]
        # result = str(pred_class[0][0][1])               # Convert to string
        # result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)