# 🌱 Plant Disease Detection Using CNN
This project implements a Convolutional Neural Network (CNN) to automatically classify and detect plant diseases from 🌿 leaf images. The model helps farmers and researchers identify diseases early, reducing crop loss and improving yield.


🚀 Features

Image-based Classification: Identifies 15 different plant diseases from a simple photo of a leaf.

High Accuracy: The model is fine-tuned on a comprehensive dataset to achieve high performance.

User-Friendly Interface: The Streamlit web app provides a simple and intuitive platform for users to get predictions.

Robust Training: The model incorporates advanced techniques like data augmentation and dropout to prevent overfitting and ensure reliable predictions on new, unseen data.


📁 Dataset and Data Preprocessing

The model was trained on the PlantVillage dataset, a well-known public dataset for plant disease classification.

Dataset Source: Kaggle - PlantVillage Dataset

Data Split: The dataset was divided into:

Training Set: 16,504 images

Validation Set: 2,058 images

Test Set: 2,076 images

Image Preprocessing: All images were resized to 128x128 pixels.

Data Augmentation: To improve the model's ability to generalize, the training data was augmented with random flips, rotations, and zooms.


🧠 Model Architecture

The model architecture combines a powerful pre-trained model with a custom classification head.

Base Model: MobileNetV2, pre-trained on ImageNet. The last 100 layers of this model were unfrozen and fine-tuned on our dataset.

Classification Head: A custom-built classifier was added on top of the base model's output. It consists of:

A GlobalAveragePooling2D layer to flatten the feature maps.

A Dense layer with 512 neurons.

A Dropout layer (rate = 0.6) for regularization.

A final Dense layer with a softmax activation for multi-class prediction.

🚀 How to Run the Project Locally
To run this project, you'll need a Python environment with the required libraries.

Prerequisites

Python (3.7+)

The required libraries are listed in the requirements.txt file.

Clone the repository:

git clone [https://github.com/](https://github.com/)[Your GitHub Username]/[Your Repository Name].git
cd [Your Repository Name]

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:
Ensure your trained model file leaf_disease_model_stable.keras is in the same directory as app.py. Then, execute:

streamlit run app.py
