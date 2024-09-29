import pandas as pd
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('breast_cancer_data.csv') 
X = data.drop('target', axis=1)
y = data['target']

# Train-test split and preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualization
shap.summary_plot(shap_values, X_test)

# LIME
lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=X.columns, class_names=['Not Cancer', 'Cancer'], mode='classification')
i = 0  # Index of the instance to explain
exp = lime_explainer.explain_instance(X_test[i], model.predict_proba, num_features=5)
exp.show_in_notebook(show_table=True)
