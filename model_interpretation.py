import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
data = pd.read_csv('breast_cancer_data.csv')  
X = data.drop('target', axis=1)
y = data['target']

# Train-test split and preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Load the trained model
model = joblib.load('random_forest_model.pkl') 

# SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=X.columns)
plt.title("SHAP Summary Plot")
plt.savefig('shap_summary_plot.png')  # Save the plot as a file
plt.show()

# SHAP dependence plot for a specific feature (e.g., 'feature1')
feature_index = X.columns.get_loc('feature1')  # Replace 'feature1' with your feature name
plt.figure()
shap.dependence_plot(feature_index, shap_values[1], X_test, feature_names=X.columns)
plt.title("SHAP Dependence Plot for Feature1")
plt.savefig('shap_dependence_plot_feature1.png')  # Save the plot as a file
plt.show()

# LIME explanation
lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=X.columns, class_names=['Not Cancer', 'Cancer'], mode='classification')
i = 0  # Index of the instance to explain
exp = lime_explainer.explain_instance(X_test[i], model.predict_proba, num_features=5)

# LIME visualization
fig = exp.as_pyplot_figure()
plt.title("LIME Explanation for Test Instance")
plt.savefig('lime_explanation_plot.png')  # Save the LIME plot as a file
plt.show()

# Display feature importance from Random Forest model
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]
plt.figure()
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importance[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.savefig('feature_importances_plot.png')  # Save the feature importance plot as a file
plt.show()

print("SHAP and LIME analyses complete. Visualizations saved as PNG files.")
