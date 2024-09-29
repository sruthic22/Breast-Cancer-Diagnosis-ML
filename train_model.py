import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import joblib

# Load dataset
data = pd.read_csv('breast_cancer_data.csv') 

# Data preprocessing
# Handling missing values
data.fillna(data.mean(), inplace=True)  # Replace NaN values with mean

# Feature selection (for illustration, adjust based on your dataset)
features = data.drop('target', axis=1)  # Features
target = data['target']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% variance
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Model selection and hyperparameter tuning
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Hyperparameter grid for Random Forest
param_grid = {
    'Random Forest': {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
}

best_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    
    # If model is Random Forest, apply GridSearchCV for hyperparameter tuning
    if name == 'Random Forest':
        grid_search = GridSearchCV(model, param_grid[name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        model.fit(X_train, y_train)
        best_model = model
    
    # Save the best model for future use
    joblib.dump(best_model, f'{name.lower().replace(" ", "_")}_model.pkl')

    # Model evaluation
    y_pred = best_model.predict(X_test)
    print(f"{name} Model Evaluation:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
    
    best_models[name] = best_model

# Save scaler and PCA components for future use
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')

print("Model training and evaluation completed. Models and preprocessing objects have been saved.")
