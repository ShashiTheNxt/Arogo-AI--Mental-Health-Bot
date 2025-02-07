import streamlit as st
import joblib
import numpy as np
import pandas as pd
import lime.lime_tabular
import os

# Initialize session state to store model and encoders
if 'loaded' not in st.session_state:
    # Load the trained model and encoders
    st.session_state.xgb_model = joblib.load("/Users/shashikanthbandoju/Desktop/AROGOAI TASK/code/models/depression_anxiety_data.csv_models/xgboost_model.pkl")
    st.session_state.label_encoders = joblib.load("/Users/shashikanthbandoju/Desktop/AROGOAI TASK/code/encoders/depression_anxiety_data.csv_encoders/label_encoders.pkl")
    st.session_state.target_encoder = joblib.load("/Users/shashikanthbandoju/Desktop/AROGOAI TASK/code/encoders/depression_anxiety_data.csv_encoders/target_encoder.pkl")
    st.session_state.X_train = joblib.load("/Users/shashikanthbandoju/Desktop/AROGOAI TASK/code/datasets/datasets_pickle/depression_anxiety_data.csv_pickle/X_train.pkl")

    # Initialize LIME explainer
    st.session_state.explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(st.session_state.X_train),
        feature_names=st.session_state.X_train.columns.tolist(),
        class_names=st.session_state.target_encoder.classes_,
        mode="classification",
        discretize_continuous=False
    )
    st.session_state.loaded = True

# Streamlit UI
st.title("Mental Health Prediction & Explainability")
st.write("Enter details below to predict mental health condition and generate LIME explanations.")

# User input fields
age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", ["male", "female",])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=24.5)
who_bmi = st.selectbox("WHO BMI Category", ["Underweight", "Normal", "Overweight",])
phq_score = st.number_input("PHQ Score", min_value=0, max_value=27, value=12)
depressiveness = st.checkbox("Depressiveness", value=True)
suicidal = st.checkbox("Suicidal", value=False)
depression_diagnosis = st.checkbox("Depression Diagnosis", value=False)
depression_treatment = st.checkbox("Depression Treatment", value=False)
gad_score = st.number_input("GAD Score", min_value=0, max_value=21, value=10)
anxiousness = st.checkbox("Anxiousness", value=True)
anxiety_diagnosis = st.checkbox("Anxiety Diagnosis", value=False)
anxiety_treatment = st.checkbox("Anxiety Treatment", value=False)
epworth_score = st.number_input("Epworth Score", min_value=0, max_value=24, value=6)
sleepiness = st.checkbox("Sleepiness", value=False)

# Convert user input to DataFrame
user_input = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "bmi": bmi,
    "who_bmi": who_bmi,
    "phq_score": phq_score,
    "depressiveness": depressiveness,
    "suicidal": suicidal,
    "depression_diagnosis": depression_diagnosis,
    "depression_treatment": depression_treatment,
    "gad_score": gad_score,
    "anxiousness": anxiousness,
    "anxiety_diagnosis": anxiety_diagnosis,
    "anxiety_treatment": anxiety_treatment,
    "epworth_score": epworth_score,
    "sleepiness": sleepiness
}])

# Encode categorical features
for col in st.session_state.label_encoders:
    user_input[col] = st.session_state.label_encoders[col].transform(user_input[col])

# Perform prediction
if st.button("Predict & Explain"):
    instance = user_input.iloc[0].values.reshape(1, -1)
    
    # Model prediction
    prediction = st.session_state.xgb_model.predict(instance)
    predicted_label = st.session_state.target_encoder.inverse_transform(prediction)[0]

    # Display prediction result
    st.success(f"**Predicted Mental Health Condition:** {predicted_label}")

    # Generate LIME explanation
    st.write("### LIME Explanation:")
    exp = st.session_state.explainer.explain_instance(instance[0], st.session_state.xgb_model.predict_proba)
    
    # Save LIME output to an HTML file
    lime_html_path = "lime_explanation.html"
    exp.save_to_file(lime_html_path)

    # Display download button for LIME explanation
    with open(lime_html_path, "r") as f:
        lime_html = f.read()
        st.download_button("Download LIME Explanation", lime_html, "lime_explanation.html", "text/html")

    # Remove temp file
    os.remove(lime_html_path)

    st.write("Open the downloaded file to view the LIME explanation.")