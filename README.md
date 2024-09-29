# Breast-Cancer-Diagnosis-ML

### Description:
Developed and implemented traditional and deep learning models to classify breast cancer tumors with high accuracy, utilizing Logistic Regression, Decision Trees, SVM, and neural networks. Employed hyperparameter tuning and dimensionality reduction techniques like PCA and t-SNE. Built a FastAPI web application for real-time predictions of malignancy and recurrence risk.

### Note on Confidentiality:
This project was part of a research initiative at Northeastern University focused on breast cancer prediction and analysis. Due to the sensitive nature of the research, not all code and data can be shared publicly. The files included in this repository represent selected portions of our work and demonstrate the methodologies and techniques employed during the project. Thank you for your understanding.

---

### 1. **train_model.py**
This script handles the preprocessing of the breast cancer dataset and trains a machine learning model using traditional algorithms like Random Forest. It performs data scaling, dimensionality reduction with PCA, and evaluates model performance using classification metrics. This foundational model serves as the basis for further analysis and prediction tasks.

### 2. **model_interpretation.py**
This script implements advanced techniques for model interpretation and visualization, utilizing SHAP and LIME to explain the predictions made by the machine learning model. It provides insights into feature importance and allows for a deeper understanding of how different attributes affect predictions, ensuring transparency in the predictive analytics process.

### 3. **app.py**
This FastAPI web application enables real-time predictions of tumor malignancy and recurrence risk based on user-inputted features. It loads a pre-trained machine learning model and scaler to provide accessible predictions, enhancing the usability of the predictive analytics developed in this project. Users can interact with the model through a simple RESTful API.

