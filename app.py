import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the .pkl files
with open("best_model.pkl", "rb") as model_file:
    best_model_name = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("labelencoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)

# Streamlit app
def main():
    st.title("Pancreatic Cancer Prediction App")
    st.write("Provide the input data for prediction")

    # User input fields
    sample_origin = st.selectbox("Sample Origin", ["Blood", "Tissue", "Other"])  # Adjust categories
    age = st.number_input("Age", min_value=0, step=1)
    sex = st.selectbox("Sex", ["M", "F"])
    stage = st.selectbox("Stage", ["I", "II", "III", "IV", "Unknown"])
    benign_sample_diagnosis = st.selectbox("Benign Sample Diagnosis", ["Yes", "No", "Unknown"])
    plasma_CA19_9 = st.number_input("Plasma CA19_9", step=0.01)
    creatinine = st.number_input("Creatinine", step=0.0001)
    LYVE1 = st.number_input("LYVE1", step=0.01)
    REG1B = st.number_input("REG1B", step=0.01)
    TFF1 = st.number_input("TFF1", step=0.01)
    REG1A = st.number_input("REG1A", step=0.01)

    # Prepare input data
    if st.button("Predict"):
        try:
            # Create DataFrame
            input_data = pd.DataFrame({
                "sample_origin": [sample_origin],
                "age": [age],
                "sex": [sex],
                "stage": [stage],
                "benign_sample_diagnosis": [benign_sample_diagnosis],
                "plasma_CA19_9": [plasma_CA19_9],
                "creatinine": [creatinine],
                "LYVE1": [LYVE1],
                "REG1B": [REG1B],
                "TFF1": [TFF1],
                "REG1A": [REG1A],
            })

            # Encode categorical variables (Ensure it matches training)
            input_data["sex"] = label_encoder.transform(input_data["sex"])

            # Scale numerical features
            numerical_features = ["age", "plasma_CA19_9", "creatinine", "LYVE1", "REG1B", "TFF1", "REG1A"]
            input_data[numerical_features] = scaler.transform(input_data[numerical_features])

            # Predict
            prediction = best_model_name.predict(input_data)

            st.success(f"Prediction: {prediction[0]}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
