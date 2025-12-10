## Project Overview

This project demonstrates a clinical safety analytics pipeline built using real medication error data from 2021–2024. It includes data preparation, exploratory analysis, risk pattern identification, statistical testing, and forecasting to support clinical decision-making and safety improvement initiatives. This work reflects analytical tasks aligned with a Clinical Safety Data Analyst role, including reviewing trends, interpreting risk drivers, and translating findings into operational insight.

## Key Artifacts in This Project

- Executive Summary (PDF)
- Full Python Notebook (Jupyter .ipynb)
- Original Medication Error Dataset (Excel source)
- Statistical Flow Process (as described in the notebook)

## Analysis Notebook

You can view the full analysis here: [GMR_Med_Error_Jan_21_to_Jul_24.ipynb](GMR_Med_Error_Jan_21_to_Jul_24.ipynb)

## Statistical Testing Framework (Guided by Flowchart)

This project follows a formal statistical decision framework to choose the correct test based on data type and study design.  
Examples include:

- Comparing medication error rates across certificates → Chi-Square Test  
- Evaluating differences in dosing accuracy between years → Two-Sample t-Test  
- Testing variance stability over time → F-Test  
- Identifying predictors of severe outcomes → Decision Tree Classifier  


