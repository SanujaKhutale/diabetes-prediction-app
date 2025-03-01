# app.py
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os
from sklearn.metrics import roc_curve, auc  # Add this import

# Load the model and preprocessor
model = joblib.load("diabetes_xgboost_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Streamlit app
st.title("Diabetes Prediction App")

# Input fields
st.sidebar.header("User Input Features")

# Allow users to type their details
high_bp = st.sidebar.number_input("High Blood Pressure (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0)
high_chol = st.sidebar.number_input("High Cholesterol (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0)
chol_check = st.sidebar.number_input("Cholesterol Check (0 = No, 1 = Yes)", min_value=0, max_value=1, value=1)
bmi = st.sidebar.number_input("BMI (10-50)", min_value=10, max_value=50, value=25)
smoker = st.sidebar.number_input("Smoker (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0)
stroke = st.sidebar.number_input("Stroke (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0)
heart_disease = st.sidebar.number_input("Heart Disease or Attack (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0)
phys_activity = st.sidebar.number_input("Physical Activity (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0)
fruits = st.sidebar.number_input("Fruits (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0)
veggies = st.sidebar.number_input("Veggies (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0)
hvy_alcohol_consump = st.sidebar.number_input("Heavy Alcohol Consumption (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0)
any_healthcare = st.sidebar.number_input("Any Healthcare (0 = No, 1 = Yes)", min_value=0, max_value=1, value=1)
no_doc_bc_cost = st.sidebar.number_input("No Doctor Due to Cost (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0)
gen_hlth = st.sidebar.number_input("General Health (1 = Excellent, 5 = Poor)", min_value=1, max_value=5, value=3)
ment_hlth = st.sidebar.number_input("Mental Health (0-30 days)", min_value=0, max_value=30, value=0)
phys_hlth = st.sidebar.number_input("Physical Health (0-30 days)", min_value=0, max_value=30, value=0)
diff_walk = st.sidebar.number_input("Difficulty Walking (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0)
sex = st.sidebar.number_input("Sex (0 = Female, 1 = Male)", min_value=0, max_value=1, value=0)
age = st.sidebar.number_input("Age (18-100)", min_value=18, max_value=100, value=30)
education = st.sidebar.number_input("Education (1-6)", min_value=1, max_value=6, value=1)
income = st.sidebar.number_input("Income (1-8)", min_value=1, max_value=8, value=1)

# Create input DataFrame
input_data = pd.DataFrame({
    'HighBP': [high_bp],
    'HighChol': [high_chol],
    'CholCheck': [chol_check],
    'BMI': [bmi],
    'Smoker': [smoker],
    'Stroke': [stroke],
    'HeartDiseaseorAttack': [heart_disease],
    'PhysActivity': [phys_activity],
    'Fruits': [fruits],
    'Veggies': [veggies],
    'HvyAlcoholConsump': [hvy_alcohol_consump],
    'AnyHealthcare': [any_healthcare],
    'NoDocbcCost': [no_doc_bc_cost],
    'GenHlth': [gen_hlth],
    'MentHlth': [ment_hlth],
    'PhysHlth': [phys_hlth],
    'DiffWalk': [diff_walk],
    'Sex': [sex],
    'Age': [age],
    'Education': [education],
    'Income': [income]
})

# Add Age_Group and BMI_Category (consistent with training data)
input_data['Age_Group'] = pd.cut(input_data['Age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle-aged', 'Elderly'])
input_data['BMI_Category'] = pd.cut(input_data['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# Calculate Health_Index (consistent with training data)
input_data['Health_Index'] = input_data[['GenHlth', 'MentHlth', 'PhysHlth']].mean(axis=1)

# Display Input Summary
st.subheader("Input Summary")
st.write(input_data)

# Add a Reset Button
if st.sidebar.button("Reset"):
    st.experimental_rerun()  # Reload the app to reset inputs

# Add a Submit Button
if st.sidebar.button("Predict"):
    # Preprocess input data
    input_data_transformed = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_transformed)
    prediction_proba = model.predict_proba(input_data_transformed)

    # Display results
    st.subheader("Prediction")
    st.write("Diabetes Risk:" , "High" if prediction[0] == 1 else "Low")
    st.write("Prediction Probability:", prediction_proba)

    # Lifestyle Recommendations
    st.subheader("Lifestyle Recommendations")
    if prediction[0] == 1:
        st.write("""
        - **Diet**: Reduce sugar and carbohydrate intake. Focus on whole grains, lean proteins, and vegetables.
        - **Exercise**: Aim for at least 30 minutes of moderate exercise (e.g., walking, cycling) 5 days a week.
        - **Monitoring**: Regularly check your blood sugar levels and consult a doctor.
        """)
    else:
        st.write("""
        - **Diet**: Maintain a balanced diet with fruits, vegetables, and whole grains.
        - **Exercise**: Engage in regular physical activity to stay healthy.
        - **Prevention**: Avoid smoking and limit alcohol consumption.
        """)

    # SHAP Summary Plot
    st.subheader("SHAP Summary Plot")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data_transformed)
    shap.summary_plot(shap_values, input_data_transformed, feature_names=preprocessor.get_feature_names_out(), show=False)
    st.pyplot(plt.gcf())
    plt.clf()  # Clear the plot

    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, thresholds = roc_curve([prediction[0]], prediction_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt.gcf())
    plt.clf()  # Clear the plot

    # Save User History
    st.subheader("Save Your Results")
    user_name = st.text_input("Enter your name to save your results:")
    if st.button("Save"):
        if user_name:
            # Save input and prediction to a CSV file
            user_history = input_data.copy()
            user_history['Diabetes_Risk'] = prediction[0]
            user_history['Prediction_Probability'] = prediction_proba[0][1]
            user_history['User_Name'] = user_name

            # Append to history file
            if not os.path.exists("user_history.csv"):
                user_history.to_csv("user_history.csv", index=False)
            else:
                user_history.to_csv("user_history.csv", mode='a', header=False, index=False)
            st.success("Your results have been saved!")
        else:
            st.error("Please enter your name to save your results.")

# Display User History
if st.sidebar.button("View History"):
    if os.path.exists("user_history.csv"):
        history_df = pd.read_csv("user_history.csv")
        st.subheader("User History")
        st.write(history_df)
    else:
        st.warning("No history found.")