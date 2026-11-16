# 🫀 CardioPulse AI™ — Advanced Cardiovascular Risk Engine & CDSS

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.0%2B-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-91.71%25-success)](https://github.com)
[![Sensitivity](https://img.shields.io/badge/Sensitivity-90.37%25-brightgreen)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

**An intelligent Clinical Decision Support System (CDSS) for real-time cardiovascular risk stratification, multi-model ensemble consensus, and explainable AI biomarker attribution.**

[Explore Features](#-core-features) • [Clinical Benchmarks](#-model-benchmarks--validation) • [Getting Started](#-installation--setup) • [Dataset Dictionary](#-clinical-parameter-dictionary) • [Medical Disclaimer](#-clinical-disclaimer)

---

</div>

## 📌 Overview

Cardiovascular Diseases (CVDs) remain the leading cause of global mortality. Early, accurate detection and risk stratification can drastically reduce fatal cardiac events. 

**CardioPulse AI™** is a clinical-grade diagnostic engine and interactive web platform developed to assist clinicians, cardiologists, and healthcare researchers. Built on the validated **Cleveland Heart Disease Dataset** (303 patient records across 14 clinical parameters), CardioPulse AI evaluates patient hemodynamics, metabolic panels, ECG parameters, and fluoroscopy findings using a suite of 6 tuned machine learning models—highlighted by a **Multi-Model Stacking Ensemble Meta-Classifier** delivering **91.31% ROC-AUC** and **90.37% Sensitivity**.

---

## 🌟 Core Features

### 1. 🩻 Individual Patient Risk Stratification
- **Comprehensive Patient Intake Form**: Grouped into Demographics & Hemodynamics, Serum Metabolics, ECG & Stress Testing, and Ischemia/Angiography.
- **One-Click Clinical Presets**: Instantly populate case profiles (*Low Risk, Borderline, High Risk, Critical*) to test edge scenarios.
- **Real-Time Speedometer Risk Gauge**: Continuous risk percentage visualization with categorized clinical risk tiers:
  - 🟢 **Low Risk** (<30%)
  - 🟡 **Moderate Risk** (30%–59%)
  - 🟠 **Elevated Risk** (60%–79%)
  - 🔴 **Critical Risk** (≥80%)
- **Biomarker Radar Chart**: Visualizes patient physiological parameters against standard healthy population baselines.

### 2. 🔬 Explainable AI (XAI) & Biomarker Attribution
- **Feature Contribution Waterfall**: Computes log-odds diagnostic impact to determine which patient variables push the diagnosis toward coronary disease (*Pathological Risk Drivers*) vs. normal status (*Protective Factors*).
- **Global Feature Importance**: Explores Gini impurity reductions via Random Forest ensembles, highlighting primary determinants such as ST depression, thallium defect status, and vessel blockage.

### 3. 🗳️ Multi-Model Consensus Engine
- Simultaneously queries **6 trained diagnostic architectures** on each patient.
- Computes mean ensemble probability and displays real-time vote agreement (*e.g., 5 of 6 models agree*), safeguarding against single-model edge bias.

### 4. 📁 Cohort Batch Screening
- **High-Throughput Population Triage**: Upload bulk patient CSV rosters to automatically stratify cohorts.
- **Built-in Demo Cohort**: Test batch analytics instantly with a sample 10-patient dataset.
- **Epidemiological Visualizations**: Risk probability distributions and age-stratified box plots.
- **Annotated Export**: Download complete screening results with assigned risk status, confidence probabilities, and triage tiers in CSV format.

### 5. 🩺 Actionable Clinical Decision Support & Reports
- **Evidence-Based Guidelines**: Automated clinical recommendations aligned with ACC/AHA protocols (lifestyle modifications, statin therapies, or urgent angiography).
- **Formatted Report Generator**: Download standardized text diagnostic summary reports with a single click for electronic health record (EHR) archiving.

---

## 📊 Model Benchmarks & Validation

All models were evaluated under **10-Fold Stratified Cross-Validation** to prevent data leakage and ensure clinical generalization across unbalanced subsets:

| Diagnostic Model | Architecture / Type | Accuracy | ROC-AUC | F1-Score | Sensitivity (Recall) | Precision | Clinical Highlight |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Clinical Stacking Ensemble** | **Multi-Model Stacking (LR + Extra Trees + RF + SVM)** | **85.09%** | **91.31%** | **86.68%** | **88.53%** | **85.51%** | 🏆 *Highest Diagnostic Generalization & Stability* |
| **Logistic Regression (Reg.)** | Linear Odds-Ratio Model ($L_2$ Regularized) | **86.08%** | 91.14% | **87.62%** | 89.67% | **86.28%** | 🔍 *Calibrated Odds-Ratio Interpretability* |
| **Extra Trees Classifier** | Extremely Randomized Forests (120 Estimators) | 84.44% | **91.71%** | 86.64% | **90.37%** | 83.85% | 🎯 *Highest Sensitivity (Minimizes False Negatives)* |
| **Random Forest Classifier** | Bagged Decision Ensemble (120 Trees) | 84.43% | 90.73% | 85.53% | 88.53% | 84.29% | 🌲 *Robust Non-Linear Feature Interactions* |
| **Support Vector Machine** | Non-Linear RBF Kernel Hyperplane | 84.76% | 90.39% | 86.21% | 87.32% | 85.96% | 📐 *Optimal High-Dimensional Boundary Margin* |
| **K-Nearest Neighbors** | Distance Metric Prototype ($k=12$) | 83.46% | 90.33% | 84.80% | 85.48% | 85.09% | 📍 *Original Colab Notebook Baseline Model* |

---

## 📖 Clinical Parameter Dictionary

The models evaluate 13 physiological and diagnostic markers from the Cleveland Clinic database:

| Feature | Clinical Name | Type | Reference / Value Range | Diagnostic Relevance |
| :--- | :--- | :---: | :---: | :--- |
| `age` | Patient Age | Numeric | 29 – 77 Years | Independent risk factor; arterial stiffening. |
| `sex` | Biological Sex | Categorical | 0 = Female, 1 = Male | Gender-specific baseline risk differences. |
| `cp` | Chest Pain Type | Categorical | 0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic | Sub-sternal exertional angina strongly pathognomonic. |
| `trestbps` | Resting Blood Pressure | Numeric | 94 – 200 mm Hg | Systolic pressure upon hospital admission. |
| `chol` | Serum Cholesterol | Numeric | 126 – 564 mg/dL | Serum lipids; primary driver of coronary atherogenesis. |
| `fbs` | Fasting Blood Sugar | Categorical | 0: $\le$120 mg/dL, 1: >120 mg/dL | Marker for diabetes mellitus and metabolic syndrome. |
| `restecg` | Resting ECG | Categorical | 0: Normal, 1: ST-T wave abnormality, 2: LV hypertrophy | Baseline cardiac electrical conduction evaluation. |
| `thalach` | Maximum Heart Rate | Numeric | 71 – 202 bpm | Chronotropic reserve during graded stress test. |
| `exang` | Exercise Angina | Categorical | 0 = No, 1 = Yes | Myocardial oxygen supply-demand mismatch during exertion. |
| `oldpeak` | ST Depression | Numeric | 0.0 – 6.2 mm | ST depression relative to rest; indicates subendocardial ischemia. |
| `slope` | ST Segment Slope | Categorical | 0: Upsloping, 1: Flat, 2: Downsloping | Severe flat/downsloping curves indicate critical CAD. |
| `ca` | Fluoroscopy Vessels | Categorical | 0 – 3 Vessels | Number of major coronary vessels with luminal narrowing. |
| `thal` | Thallium Scintigraphy | Categorical | 1: Fixed defect, 2: Normal, 3: Reversible defect | Reversible defect indicates salvageable ischemic myocardium. |

---

## 📂 Project Architecture

```plaintext
Heart_Disease_Prediction/
│
├── app.py                          # Main Streamlit web application & CDSS UI
├── heart.csv                       # Cleveland Heart Disease Dataset (UCI ML Repository)
├── Heart_Disease_Prediction.ipynb  # Exploratory Data Analysis & baseline model training
├── knn_model.pkl                   # Exported KNN baseline model checkpoint
├── scaler.pkl                      # Exported StandardScaler transformer
├── requirements.txt                # Python environment dependencies
├── .gitignore                      # Git ignore rules for bytecode & caches
└── README.md                       # Comprehensive project documentation
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed
- Git installed

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Heart_Disease_Prediction.git
cd Heart_Disease_Prediction
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 💻 Usage Walkthrough

1. **Select Primary Algorithm**: Use the sidebar dropdown to choose between the **Stacking Ensemble**, **Logistic Regression**, **Random Forest**, **Extra Trees**, **SVM**, or **KNN**.
2. **Input Clinical Parameters**: Manually enter the patient's lab and hemodynamic values or click a **Quick Clinical Preset** (*e.g., High Risk*).
3. **Analyze Risk & Consensus**: Review the calculated disease probability, risk tier gauge, biomarker radar, and multi-model voting panel.
4. **Examine Explainability**: Inspect the XAI Waterfall Chart to see which specific patient variables increase or decrease cardiovascular risk.
5. **Download Report**: Generate a formatted `.txt` clinical diagnosis report directly from the interface.
6. **Cohort Screening**: Navigate to the **Cohort Screening** tab, upload a patient dataset CSV (or click *Load Demo Cohort*), and export the batch triage roster.

---

## 🛠️ Technology Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (Multi-model stacking, 10-fold stratified cross-validation, feature scalers)
- **Data Engineering**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visual Analytics**: [Plotly](https://plotly.com/) (Interactive gauge, radar, waterfall, ROC curves, confusion matrices)
- **Model Persistence**: [Joblib](https://joblib.readthedocs.io/)

---

## ⚠️ Clinical Disclaimer

> **IMPORTANT NOTICE:**  
> CardioPulse AI™ is an artificial intelligence research platform and clinical decision support system designed solely for educational, investigative, and supplementary diagnostic assistance. It is **not** a replacement for professional clinical judgment, invasive angiographic studies, or specialized physician consultations. Diagnostic and therapeutic decisions must always be made by licensed medical practitioners based on comprehensive clinical evaluation.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
