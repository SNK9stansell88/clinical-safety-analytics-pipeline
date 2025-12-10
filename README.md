# Clinical Safety Analytics Pipeline – Medication Error Analysis (2021–2024)

This repository showcases a clinical safety analytics project focused on **medication-related safety events** within an air medical / EMS environment. It is designed as a **portfolio-ready project** demonstrating how medication error data can be structured, analyzed, and communicated to support a **Clinical Safety Data Analyst** role.

The core objective is to turn raw medication error reports into **actionable safety insight** that can inform education, protocol refinement, and future AI-driven decision support.

---

## 1. Project Overview

This project analyzes medication-related safety reports from January 2021 through July 2024. Using a structured Excel dataset and a reproducible notebook, the work:

- Consolidates medication error events from multiple branches and certificates
- Normalizes date fields into a true event date
- Summarizes where, when, and how medication events occur
- Highlights patterns in **Risk Event**, **Outcome**, and **Medication** fields
- Prepares a cleaned export suitable for dashboards, additional statistics, or model development
- Connects the technical analysis to **clinical safety questions** (e.g., dosing errors, wrong drug, adverse effects)

The analysis is intended to support future work such as:
- Root-cause analysis (RCA)
- Quality improvement initiatives
- Early-stage AI / machine learning models for **medication error risk prediction**

---

## 2. Files in This Repository

- `GMR Med Error Jan 21 to July 24.xlsx`  
  Source Excel file containing medication-related safety events.
  - Sheet of interest: **`Medication`**  
  - Each row represents a reported medication-related event.

- `GMR_Med_Error_Jan_21_to_Jul_24_clean.ipynb`  
  Primary Jupyter notebook with the **full analysis code** (cleaned of heavy outputs so it renders on GitHub).  
  This notebook:
  - Loads the Excel dataset
  - Cleans and prepares the data
  - Builds an `Event Date`
  - Produces summary tables and visualizations
  - Explores patterns across risk, outcome, and medication type

- `Executive Summary.pdf`  
  A non-technical, leadership-facing summary of the key findings and implications from the analysis. Suitable for a Vice President of Clinical Services, QA leadership, or an interview panel.

- `Med Error 251201 1610.pdf`  
  Additional slide/report material (e.g., charts/visuals) that complements the executive summary and notebook. This can be used in presentations or as an attachment when sharing the work.

> Note: File names intentionally retain their original conventions to align with internal data sources.

---

## 3. Dataset Description

**Source:** Internal medication-related safety reports (de-identified for demonstration).

**Primary sheet used:** `Medication`

Key columns include (not exhaustive):

- `Report ID` – Unique identifier for the safety report  
- `Month`, `Day`, `Year` – Components of the event date  
- `Source` – Organization (e.g., certificate or program)  
- `Branch` – Operational branch (e.g., Air vs. Ground, region)  
- `Primary Risk` – High-level risk category (e.g., Medication)  
- `Risk Event` – Specific type of medication-related event  
- `Medication 1`, `Medication 2` – Medications involved in the event  
- `Event` – Narrative description of what occurred  
- `Medication Cross Check` – Whether a cross-check was documented  
- `Precursor/Stressor` – Factors contributing to the event  
- `Outcome` – Impact on the patient / case  
- `Pattern` – Categorized pattern type  
- `Pattern Specifics` – More detailed pattern description

These fields provide a rich basis for understanding **where medication errors cluster**, **how they present**, and **what patterns may be preventable** through better systems, training, or real-time guidance.

---

## 4. Analysis Highlights (Notebook)

The main notebook (`GMR_Med_Error_Jan_21_to_Jul_24_clean.ipynb`) is structured to be readable and reproducible. At a high level, it performs:

1. **Data Loading**
   - Reads the `Medication` sheet from the Excel file.
   - Ensures key columns are available and correctly typed.

2. **Data Cleaning**
   - Removes fully empty / placeholder columns (e.g., `Unnamed` columns).
   - Constructs a proper `Event Date` from `Day`, `Month`, `Year`.
   - Prepares a cleaned DataFrame for downstream analysis and export.

3. **Descriptive Exploration**
   - Frequency tables for:
     - `Source` and `Branch`
     - `Primary Risk` and `Risk Event`
     - `Medication 1`
     - `Outcome`, `Pattern`, `Pattern Specifics`
   - Identification of the most frequent:
     - Risk events
     - Medications involved
     - Outcomes and pattern types

4. **Temporal Trends**
   - Aggregates events by month/year (e.g., using a `YearMonth` period).
   - Visualizes the number of unique reports over time to see increases, decreases, or spikes.

5. **Cross-Tabulations**
   - Cross-tab of `Risk Event` vs. `Outcome` to understand how certain event types translate into impact.
   - Other cross-tabs can be added (e.g., by Source, Branch, Pattern).

6. **Clean Export**
   - Writes out a cleaned CSV (e.g., `med_error_clean_export.csv`) that can be:
     - Used in BI tools (Power BI, Tableau)
     - Combined with other safety datasets
     - Fed into statistical or machine learning workflows

---

## 5. How to Run This Project Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/clinical-safety-analytics-pipeline.git
   cd clinical-safety-analytics-pipeline
