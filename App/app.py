from flask import Flask, request, render_template
import os
import numpy as np
import cv2
from tensorflow import keras

app = Flask(__name__)

# Load the model
model = keras.models.load_model("rice_model.h5")

# Labels
df_labels = {
    0: 'Arborio',
    1: 'Basmati',
    2: 'Ipsala',
    3: 'Jasmine',
    4: 'Karacadag'
}

# Home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')


# Prediction route
@app.route('/result', methods=['POST'])
def result():
    if 'image' not in request.files:
        return "No image uploaded", 400

    f = request.files['image']
    if f.filename == '':
        return "Empty filename", 400

    basepath = os.path.dirname(__file__)
    upload_path = os.path.join(basepath, 'uploads')
    os.makedirs(upload_path, exist_ok=True)
    filepath = os.path.join(upload_path, f.filename)
    f.save(filepath)

    # Preprocess image
    img = cv2.imread(filepath)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    pred = model.predict(img)
    pred_class = np.argmax(pred)

    prediction_text = df_labels.get(pred_class, "Unknown")

    return render_template('result.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
