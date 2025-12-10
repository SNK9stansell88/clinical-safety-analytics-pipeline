# Clinical Safety Analytics Pipeline – Medication Error Analysis (2021–2024)

This repository showcases a clinical safety analytics project focused on **medication-related safety events** within an air medical / EMS environment. It demonstrates how raw medication error data can be transformed into actionable safety insight to support operational decision-making, quality improvement, and future AI-driven clinical decision support systems.

The core objective is to turn de-identified medication error records into **structured safety intelligence** that supports education, risk mitigation, and long-term predictive modeling.

---

## 1. Project Overview

This project analyzes medication-related safety reports from **January 2021 through July 2024** across multiple organizational certificates. The goals are to:

- Consolidate and standardize medication error events  
- Clean and normalize date fields into a true event timeline  
- Identify trends by branch, risk category, outcome, and medication type  
- Explore patterns in **Primary Risk**, **Risk Event**, **Outcome**, and **Medication** fields  
- Produce a clean dataset for dashboards, statistical analysis, or machine learning  
- Demonstrate how clinical and data-driven analysis work together to improve patient safety  

The results support:

- Root cause analysis (RCA)  
- Targeted safety interventions  
- Clinical education and protocol alignment  
- Early-stage development of **AI-powered risk prediction tools**  

---

## 2. Repository Structure

clinical-safety-analytics-pipeline/
│
├── README.md
├── Executive Summary.pdf
├── GMR Med Error Jan 21 to July 24.xlsx
│
└── analysis_notebooks/
    ├── GMR_Med_Error_Jan_21_to_Jul_24_script.py   ← main analysis notebook (script format)
    └── .gitkeep

---

## 3. File Descriptions

### 📘 GMR Med Error Jan 21 to July 24.xlsx
Primary source file containing medication-related safety reports.  
Sheet of interest: **Medication**.  
Each row represents a reported medication event.

---

### 📓 analysis_notebooks/GMR_Med_Error_Jan_21_to_Jul_24_script.py
This is the **cleaned, complete analysis notebook converted to a Python script**.  
GitHub cannot render large `.ipynb` notebooks with heavy output, so this `.py` version ensures full visibility.

This script includes:

- Data loading & validation  
- Removal of placeholder/unnamed columns  
- Construction of a true `Event Date`  
- Summary statistics  
- Trend analysis  
- Frequency tables (Medication, Risk Event, Outcome, Pattern)  
- Pattern analysis aligned with clinical interpretation  

This is the **core analytical file** for the project.

---

### 📄 Executive Summary.pdf
A concise, leadership-facing summary designed for Clinical Services executives, QA leaders, and interview panels.  
Highlights major findings, trends, and operational implications.

---

## 4. Dataset Description

Key fields include:

- **Source** – Organizational certificate  
- **Branch** – Program/location  
- **Primary Risk** – High-level risk category  
- **Risk Event** – Specific medication error type  
- **Medication 1 / Medication 2** – Drug(s) involved  
- **Event Description** – Narrative description  
- **Precursor/Stressor** – Contributing factors  
- **Outcome** – Patient impact  
- **Pattern / Pattern Specifics** – Categorized error pattern  
- **Month / Day / Year** – Used to create a real `Event Date`  

These fields allow detailed exploration of **how**, **where**, and **why** medication-related safety issues occur.

---

## 5. Analysis Highlights

### ✔ Data Cleaning
- Removes empty/unnamed columns  
- Standardizes field names  
- Creates `Event Date`  
- Handles missing or malformed data  

### ✔ Descriptive Statistics
- Frequency of:
  - Risk Events  
  - Medications  
  - Outcomes  
  - Patterns & Pattern Specifics  
- Identifies highest-risk categories  

### ✔ Temporal Trends
- Monthly & yearly event totals  
- Spike detection  
- Emerging trends or seasonality  

### ✔ Cross-Tabulations
Example analyses include:

- Risk Event × Outcome  
- Medication × Error Pattern  
- Source × Event Frequency  

### ✔ Clean Export
A cleaned CSV export (within the script) is prepared for:

- BI dashboards (Power BI / Tableau)  
- SQL ingestion  
- Machine learning pipelines  
- QA/leadership reporting  

---

## 6. How to Run This Project Locally

### Clone the repository
```bash
git clone https://github.com/<your-username>/clinical-safety-analytics-pipeline.git
cd clinical-safety-analytics-pipeline

pip install -r requirements.txt

python analysis_notebooks/GMR_Med_Error_Jan_21_to_Jul_24_script.py
```

---

## 7. Future Directions

This project establishes a foundation for advanced analytics and AI integration, including:

### 🔹 Predictive Modeling
- XGBoost / Random Forest classification  
- Logistic regression for risk prediction  
- Time-series forecasting of medication-related safety events  

### 🔹 Clinical AI Development
- AI-assisted medication cross-check workflows  
- NLP interpretation of free-text event descriptions  
- Real-time decision support informed by:
  - Medication error risk  
  - Device data (Zoll, Hamilton, ReVel, Sapphire pumps/vents)  
  - Operational and environmental stressors  

### 🔹 System-Wide QA Improvements
- Automated dashboards integrating ImageTrend (ePCR) + Intelex (QA)  
- Heatmaps identifying high-risk medication clusters  
- Simulation models for “what-if” safety interventions  

This repository serves as the technical backbone for a future **multi-layer clinical safety analytics platform**, supporting predictive safety modeling and AI-assisted clinical guidance.

---

## 8. Contact

For collaboration, expansion of this project, or discussion of clinical safety analytics:

**Chris Stansell**  
432-559-0904  
chris.stansell@gmr.net

---




