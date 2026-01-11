# GrainPalette — A Deep Learning Odyssey in Rice Type Classification Through Transfer Learning

Overview

GrainPalette is a deep learning-based image classification project designed to automatically identify different rice grain varieties using transfer learning techniques. The solution leverages pre-trained convolutional neural networks (CNNs) to extract visual features and classify input rice grain images into distinct categories with high accuracy. This project was developed as part of an AI/ML internship and includes both model training and a web interface for real-time predictions.

Motivation

Manual classification of rice grain types based on visual characteristics like shape, size, and texture is time-consuming, requires expertise, and can be prone to errors. Automating this process using deep learning enables scalable, consistent, and rapid classification that can support agricultural stakeholders, researchers, and educators.

Features

Transfer learning using pre-trained CNN architectures for efficient feature extraction

Fine-tuning of selected models to improve classification accuracy

Trains on a dataset of rice grain images representing multiple rice varieties

Web application interface for uploading images and displaying predictions

Prediction output includes class probabilities and predicted rice type

Technologies Used

Python

TensorFlow / Keras

Transfer Learning (e.g., MobileNet, VGG, or similar pre-trained CNN models)

NumPy, Matplotlib

Flask (for web service backend)

HTML, CSS, JavaScript (frontend)

Dataset

The dataset consists of labeled rice grain images from multiple varieties. Each image contains a single rice grain sample, preprocessed and resized to standard dimensions suitable for training CNN models. The dataset may include varieties such as Basmati, Jasmine, Arborio, and others.

Model Training

Data Preprocessing

Load dataset and labels

Apply resizing, normalization, and augmentation

Transfer Learning Setup

Select a pre-trained CNN model (e.g., MobileNetV2, VGG16)

Freeze the convolutional base

Add custom fully connected layers for classification

Model Fine-Tuning

Train the added layers on the dataset

Optionally unfreeze some base layers for further tuning

Evaluation

Monitor validation metrics (accuracy, loss)

Save the best performing model

Usage
Clone the Repository
git clone https://github.com/Chaitanya2710/GrainPalette---A-Deep-Learning-Odyssey-in-Rice-Type-Classification-Through-Transfer-Learning.git

Install Dependencies

Install required Python packages:

pip install -r requirements.txt

Train the Model

Update dataset paths in the training script and run:

python train.py

Run the Web Application

Launch the Flask application:

python app.py


Open the web interface in a browser and upload rice grain images to get predictions.

Results

The model achieves robust classification performance across multiple rice grain types. Performance metrics (such as accuracy, precision, recall, and F1-score) can be monitored during evaluation to validate model quality.

GrainPalette/
│
├── data/                  # Dataset of rice grain images
├── models/                # Saved trained models
├── src/                   # Source code for training and inference
├── app.py                 # Flask backend application
├── requirements.txt       # Dependencies
├── README.md              # Documentation
└── web/                   # Frontend HTML/CSS/JS files


Future Work

Increase dataset diversity with more rice varieties

Experiment with additional transfer learning models for improved accuracy

Deploy the application using cloud hosting (e.g., AWS, GCP, or Heroku)

Add real-time camera input for live classification

License

Specify the software license under which the project is released (e.g., MIT, Apache 2.0).

References

GrainPalette documentation on project design and methodology.

LinkedIn project description and use-case context.
