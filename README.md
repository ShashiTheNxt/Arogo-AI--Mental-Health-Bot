# Arogo-AI Mental Health Bot
## **📌 Table of Contents**
- [Objective](#-objective)
- [Project Structure](#-project-structure)
- [Dataset & Preprocessing](#-dataset--preprocessing)
- [Model Development](#-model-development)
- [Model Inference](#-model-inference)
- [LLM Experimentation](#-llm-experimentation)
- [SHAP/LIME Analysis](#-shaplime-analysis)
- [Installation & Dependencies](#-installation--dependencies)
  
---

## **🎯 Objective**
Develop a **Self-Analysis Mental Health Model as called as Mental Health Bot** that predicts possible mental health conditions based on user-provided symptoms. The model should be designed for seamless integration into a chatbot or an application, with a focus on **accuracy, interpretability, and efficiency**. Additionally, a **basic UI or command-line script** should be provided for testing and interaction.

---

## **📂 Project Structure**  
📦 ArogoAI_Mental_Health_Prediction  
│  
├── 📂 datasets/                      # Datasets used for training and evaluation  
│   ├── 📂 original/                   # Raw datasets  
│   │   ├── depression_anxiety_data.csv  
│   │   ├── survey.csv (Not Used)  
│   │  
│   ├── 📂 preprocessed/               # Processed datasets  
│   │   ├── preprocessed_depression_anxiety_data.csv  
│   │   ├── preprocessed_survey.csv (Not Used)  
│   │  
│   ├── 📂 pickle/                     # Saved datasets in pickle format  
│   │   ├── depression_anxiety_data/  
│   │       ├── X_train.pkl  
│  
├── 📂 code/                           # Python scripts for training and inference  
│   ├── training.ipynb                 # Model training notebook  
│   ├── predict_mental_health.py       # CLI script for inference  
│  
├── 📂 models/                         # Trained models  
│   ├── 📂 depression_anxiety_data/  
│   │   ├── random_forest_model.pkl  
│   │   ├── xgboost_model.pkl  
│  
├── 📂 encoders/                       # Encoders for categorical variables  
│   ├── 📂 depression_anxiety_data/  
│   │   ├── label_encoders.pkl  
│   │   ├── target_encoder.pkl  
│  
├── .gitignore                         # Ignore unnecessary files  
├── requirements.txt                    # Python dependencies  
├── mental_health_report.pdf            # Documentation report  
└── README.md                           # Project documentation  

---

## **📊 Dataset & Preprocessing**
We use **publicly available mental health datasets**, including:
- **Depression and Anxiety Symptoms Dataset**
- **Mental Health in Tech Survey** (not used)

### **Preprocessing Steps**
- **Data Cleaning** – Remove inconsistencies and missing values.
- **Normalization** – Process text for better analysis.
- **Exploratory Data Analysis (EDA)** – Identify relationships between symptoms and conditions.
- **Feature Engineering** – Encode symptoms and mental health conditions as features and labels.
- **Feature Selection** – Choose the most impactful features.

Preprocessed datasets are saved in `datasets_preprocessed/`.

The code for preprocessing is present in [train.ipynb](code/train.ipynb). The encoders used in preprocessing are stored in `encoders/`.

---

## **🛠 Model Development**

We trained and compared the following models:

1. Random Forest
2. XGBoost

The best-performing model is stored in `models/depression_anxiety_data.csv_models/`.

### **Evaluation Metrics**
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- SHAP/LIME for interpretability

To train models, use:
```
python code/train.ipynb
```

---

## **🔍 Model Inference**
We provide a command-line script to test the trained model:
```
python code/predict_mental_health.py
```

---

## **🧠 LLM Experimentation**
We used LLM (Google Gemini) to:

1. Explain the predicted mental health condition in natural language.
2. Suggest coping mechanisms based on symptoms.

Results are documented in `mental_health_report.pdf`.

---

## **📈 SHAP/LIME Analysis**
To interpret model predictions, we used:
- LIME (Local Interpretable Model-Agnostic Explanations)

---

