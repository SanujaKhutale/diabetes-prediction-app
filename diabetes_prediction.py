# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import shap  # For explainable AI
import joblib  # For saving the model

# Load the dataset
df = pd.read_csv("C:/Users/Sanuja/OneDrive/Desktop/Diabetes_Prediction_Project/diabetes_dataset.csv")
print("Dataset loaded successfully.")

# Distribution of the target variable
sns.countplot(x='Diabetes_binary', data=df)
plt.title("Distribution of Diabetes (0 = No, 1 = Yes)")
plt.savefig("diabetes_distribution.png")  # Save the plot
print("Distribution plot saved as diabetes_distribution.png.")

# Feature Engineering
df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle-aged', 'Elderly'])
df['Health_Index'] = df[['GenHlth', 'MentHlth', 'PhysHlth']].mean(axis=1)
print("Feature engineering completed.")

# Data Preprocessing
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data preprocessing completed.")

# Define preprocessing for numerical and categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Add handle_unknown='ignore'
    ])

# Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
print("Data preprocessing applied.")

# Model Building
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("Model training and prediction completed.")

# Evaluation
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# SHAP explainer
try:
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, feature_names=preprocessor.get_feature_names_out(), show=False)
    plt.savefig("shap_summary.png")  # Save the plot
    print("SHAP summary plot saved as shap_summary.png.")
except Exception as e:
    print(f"SHAP error: {e}")

# ROC Curve
try:
    fpr, tpr, thresholds = roc_curve(y_test, xgb_model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (XGBoost)')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")  # Save the plot
    print("ROC Curve saved as roc_curve.png.")
except Exception as e:
    print(f"ROC Curve error: {e}")

# Save the model and preprocessor
try:
    joblib.dump(xgb_model, "diabetes_xgboost_model.pkl")
    joblib.dump(preprocessor, "preprocessor.pkl")
    print("Model and preprocessor saved successfully.")
except Exception as e:
    print(f"Error saving model/preprocessor: {e}")
