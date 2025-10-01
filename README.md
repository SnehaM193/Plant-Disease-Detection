# Plant Disease Detection Web App

# 🌱 Project Overview
This project is a web-based application designed to assist in the early and accurate detection of plant diseases. By leveraging a deep learning model, users can upload an image of a plant leaf and receive an instant diagnosis of its health, helping them take timely action to protect their crops.

The application is built using a transfer learning approach with a fine-tuned MobileNetV2 model and a front-end interface developed with Streamlit.

<br>

## 🌟Key Features

**✅ Image-based Classification**: Identifies 15 different plant diseases from a simple photo of a leaf.

**✅ High Accuracy**: The model is fine-tuned on a comprehensive dataset to achieve high performance.

**✅ User-Friendly Interface**: The **Streamli**t web app provides a simple and intuitive platform for users to get predictions.

**✅ Robust Training**: The model incorporates advanced techniques like **data augmentation** and **dropout** to prevent overfitting and ensure reliable predictions on new, unseen data. 


<br>

## 📁 Dataset and Data Preprocessing

• The model was trained on the **PlantVillage dataset**, a well-known public dataset for plant disease classification.

**• Dataset Source**: Kaggle - PlantVillage Dataset

**• Data Split**: The dataset was divided into:

   * Training Set: 16,504 images

   * Validation Set: 2,058 images

   * Test Set: 2,076 images

**• Image Preprocessing**: All images were resized to 128x128 pixels.

**• Data Augmentation**: To improve the model's ability to generalize, the training data was augmented with random flips, rotations, and zooms.


<br>

## Demo
https://plant-disease-detection-dbchgo5d6c5sadcpwy9kc7.streamlit.app/


<br>

## 🧠 Model Architecture

The model architecture combines a powerful pre-trained model with a custom classification head.

**• Base Model**: MobileNetV2, pre-trained on ImageNet. The last 100 layers of this model were unfrozen and fine-tuned on our dataset.

**• Classification Head**: A custom-built classifier was added on top of the base model's output. It consists of:

**• A GlobalAveragePooling2D**  layer, a Dense layer (512 neurons), a Dropout layer (rate = 0.6), and a final Dense layer with a **softmax** activation.

<br>

## ⚙️ Training and Evaluation

The model was trained for a maximum of 20 epochs using the Adam optimizer with a dynamic learning rate.

**Optimize**r: Adam with an initial learning rate of 0.001.

**Callbacks**:

   • **Early Stopping**: Prevents overfitting by stopping training if validation loss does not improve for 5 consecutive epochs.

   • **ReduceLROnPlateau**: Reduces the learning rate automatically when validation loss plateaus.

**Final Performance**: The model achieved a test accuracy of **98.12%** and a test loss of **0.0698** on a separate, unseen test set. This indicates excellent performance and generalization.

<br>

## 🛠️ How to Run the Project Locally

Prerequisites

• Python (3.7+)

• The required libraries are listed in the requirements.txt file.



1. Clone the repository:

```bash
git clone [https://github.com/](https://github.com/)[Your GitHub Username]/[Your Repository Name].git
cd [Your Repository Name]
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
Ensure your trained model file leaf_disease_model_stable.keras is in the same directory as app.py. Then, execute:

```bash
streamlit run app.py
```

