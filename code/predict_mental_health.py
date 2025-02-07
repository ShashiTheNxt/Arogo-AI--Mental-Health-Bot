import streamlit as st
import joblib
import numpy as np
import pandas as pd
import lime.lime_tabular
import os
import google.generativeai as genai  # Google Gemini API
from fpdf import FPDF  # PDF report generation

st.set_page_config(page_title="Mental Health Bot", layout="wide")

# Sidebar: API Key Input
st.sidebar.header("API Key Settings")
gemini_api_key = st.sidebar.text_input("Enter your Google Gemini API Key:", type="password")

if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)  # Set Gemini API key
    except Exception as e:
        st.error(f"Error configuring Gemini API key: {e}")

# Initialize session state to store model and encoders
# Initialize session state to store model and encoders
if 'loaded' not in st.session_state:
    try:
        st.session_state.xgb_model = joblib.load("/Users/shashikanthbandoju/Desktop/AROGOAI TASK/models/depression_anxiety_data.csv_models/xgboost_model.pkl")
        
        # Initialize encoders if not already in session_state
        if 'label_encoders' not in st.session_state:
            st.session_state.label_encoders = joblib.load("/Users/shashikanthbandoju/Desktop/AROGOAI TASK/encoders/depression_anxiety_data.csv_encoders/label_encoders.pkl")
        if 'target_encoder' not in st.session_state:
            st.session_state.target_encoder = joblib.load("/Users/shashikanthbandoju/Desktop/AROGOAI TASK/encoders/depression_anxiety_data.csv_encoders/target_encoder.pkl")

        st.session_state.X_train = joblib.load("/Users/shashikanthbandoju/Desktop/AROGOAI TASK/datasets/datasets_pickle/depression_anxiety_data.csv_pickle/X_train.pkl")  # Load training data for LIME

        # Initialize LIME explainer
        st.session_state.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(st.session_state.X_train),
            feature_names=st.session_state.X_train.columns.tolist(),
            class_names=st.session_state.target_encoder.classes_,
            mode="classification",
            discretize_continuous=False
        )
        st.session_state.loaded = True
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
    except Exception as e:
        st.error(f"Error loading models or encoders: {e}")


# Streamlit UI
st.title("Mental Health Bot ")
st.write("Developed by Shashikanth Bandoju for Arogo AI")

# Create two columns for input (left) and output (right)
col1, col2 = st.columns([1, 5])

# Left Column: User Inputs
with col1:
    st.subheader("User Inputs")

    # User input fields
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=24.5)
    gender = st.selectbox("Gender", ["male", "female"])
    who_bmi = st.selectbox("WHO BMI Category", ["Underweight", "Normal", "Overweight"])
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

# Function to generate explanation using Google Gemini
def generate_gemini_explanation(predicted_label, user_input_data):
    if not gemini_api_key:
        return "Please enter a valid Google Gemini API key in the sidebar."

    prompt = f"""
    Given the predicted mental health condition: {predicted_label},
    provide a natural language explanation for why this prediction was made based on the following user data:
    {user_input_data}.
    """

    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text if response else "No response generated."
    except Exception as e:
        return f"Error generating explanation: {e}"

# Function to generate a PDF report
def generate_pdf_report(predicted_label, explanation, coping_mechanisms):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Mental Health Prediction & Report", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Predicted Mental Health Condition: {predicted_label}", ln=True)
    pdf.ln(5)

    pdf.multi_cell(200, 10, txt=f"Explanation:\n{explanation}")
    pdf.ln(10)

    pdf.multi_cell(200, 10, txt=f"Coping Mechanisms and Next Steps:\n{coping_mechanisms}")

    report_path = "mental_health_report.pdf"
    pdf.output(report_path)
    return report_path

# Perform prediction and generate explanation
if st.button("Predict & Explain"):
    instance = user_input.iloc[0].values.reshape(1, -1)

    # Model prediction
    try:
        prediction = st.session_state.xgb_model.predict(instance)
        predicted_label = st.session_state.target_encoder.inverse_transform(prediction)[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")

    # Right Column: Display Model Outputs
    with col2:
        st.subheader("Model Output")

        # Display prediction result
        st.success(f"**Predicted Mental Health Condition:** {predicted_label}")

        # Generate LIME explanation
        st.write("### LIME Explanation:")
        try:
            exp = st.session_state.explainer.explain_instance(instance[0], st.session_state.xgb_model.predict_proba)
            lime_html_path = "lime_explanation.html"
            exp.save_to_file(lime_html_path)

            # Display LIME explanation as an iframe
            with open(lime_html_path, "r") as f:
                lime_html = f.read()
                st.components.v1.html(lime_html, height=400)
        except Exception as e:
            st.error(f"Error generating LIME explanation: {e}")

        # Generate LLM Explanation using Google Gemini
        user_input_str = ", ".join([f"{col}: {val}" for col, val in user_input.iloc[0].items()])
        explanation = generate_gemini_explanation(predicted_label, user_input_str)

        # Display LLM Explanation
        st.write("### Natural Language Explanation:")
        st.write(explanation)

        # Generate Coping Mechanisms using Google Gemini
        coping_mechanisms = generate_gemini_explanation(predicted_label, "Suggest coping mechanisms and next steps.")

        # Display Coping Mechanisms
        st.write("### Suggested Coping Mechanisms:")
        st.write(coping_mechanisms)

        # Generate PDF report
        report_path = generate_pdf_report(predicted_label, explanation, coping_mechanisms)

        # Download PDF Report
        with open(report_path, "rb") as f:
            st.download_button("Download PDF Report", f, "mental_health_report.pdf", "application/pdf")

        st.write("Open the downloaded file to view the report.")
