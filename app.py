import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the saved model, scaler, and label encoder
with open("best_model.pkl", "rb") as model_file:
    best_model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("labelencoder.pkl", "rb") as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

# Streamlit app UI
def main():
    st.title("Pancreatic Cancer Prediction App")
    st.write("Provide the input data for prediction")

    # Input fields for user
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

    # Prepare input data for prediction
    if st.button("Predict"):
        try:
            # Create a DataFrame from user input
            input_data = pd.DataFrame({
                "sampleBPTBorigin": [sample_origin],
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

            # Handle unseen labels: map them to "Unknown"
            input_data["sex"] = handle_unseen_labels(input_data["sex"], label_encoder)
            input_data["sampleBPTBorigin"] = handle_unseen_labels(input_data["sampleBPTBorigin"], label_encoder)
            input_data["stage"] = handle_unseen_labels(input_data["stage"], label_encoder)
            input_data["benign_sample_diagnosis"] = handle_unseen_labels(input_data["benign_sample_diagnosis"], label_encoder)

            # Standardize the input data using the saved scaler
            input_data_scaled = scaler.transform(input_data)

            # Make a prediction using the best model
            prediction = best_model.predict(input_data_scaled)

            # Display the prediction result
            st.success(f"Prediction: {'Cancer' if prediction[0] == 1 else 'No Cancer'}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

def handle_unseen_labels(column, encoder):
    """
    This function handles unseen labels by replacing them with 'Unknown'
    and encoding them using the label encoder.
    """
    # Add 'Unknown' category if it's not already present
    encoder.classes_ = np.append(encoder.classes_, "Unknown")  # Ensure 'Unknown' is part of classes_
    return encoder.transform(column.apply(lambda x: x if x in encoder.classes_ else "Unknown"))

if __name__ == "__main__":
    main()
