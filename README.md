# image-sclassifier-easy
# 🐱🐶 Cats vs Dogs Classifier (CNN Project)

A small deep learning project that classifies images as cats or dogs using Convolutional Neural Networks (CNNs).

## 📁 Dataset
- Source: [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/datasets)
- Images are preprocessed and resized to `128x128`

## 🔧 Tech Stack
- Python
- TensorFlow / Keras
- Matplotlib / Pandas / NumPy
- Jupyter Notebook

## 🧠 Model Architecture
```python
Conv2D(16, (3,3)) → MaxPooling
Conv2D(32, (3,3)) → MaxPooling
Conv2D(64, (3,3)) → MaxPooling
→ Flatten → Dense(512) → Dense(1, sigmoid)

📊 Result
Validation Accuracy: >60%

Confusion Matrix and Prediction Samples below ⬇️

🔍 Sample Predictions
<p align="center"> <img src="images/sample_prediction_dog.png" width="250"/> <img src="images/sample_prediction_cat.png" width="250"/> </p>

🧠 Future Work
Try transfer learning (e.g. MobileNetV2)

Improve accuracy with better cleaning

Deploy as web app using Streamlit or Gradio
