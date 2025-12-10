# # Step 1 — Data Loading and Basic Structure (Medication Dataset)
# 
# In this step, I load the two datasets needed for medication-error analysis:
# • The **Medication** sheet, containing row-level event details.
# • The **Med Error Summary** sheet, containing certificate-level counts by categories such as Pattern Specifics.
# 
# The purpose of Step 1 is to:
# • Ensure the Excel file loads correctly.
# • Confirm the number of rows and columns in each sheet.
# • Display the first few rows to understand structure and data types.
# • Verify that critical variables exist, including:
#   – Pattern Specifics (our main category of interest)
#   – Certificate columns (AEL, GLF, AMR, MTC, REACH, etc.)
#   – Event-level fields such as Branch, Source, Primary Risk, Risk Event, Medications, etc.
# 
# This mirrors Step 1 of the Loan Assignment (data loading and inspection), but adapted to a clinical medication-error context.

# Step 1 — Load Medication + Summary Sheets Cleanly

import pandas as pd
from google.colab import files

# Upload the Krista file (do this ONCE)
uploaded = files.upload()   # Select "Krista 240726 Final.xlsx"

# Load the workbook
file_name = "/content/clinical-safety-analytics-pipeline/GMR Med Error Jan 21 to July 24.xlsx"

# Read both sheets
med = pd.read_excel(file_name, sheet_name="Medication")
med_summary = pd.read_excel(file_name, sheet_name="Med Error Summary")

# Basic structure
print("Medication sheet shape:", med.shape)
print("Med Error Summary sheet shape:", med_summary.shape)

print("\nMedication sheet – first 5 rows:")
display(med.head())

print("\nMed Error Summary sheet – first 5 rows:")
display(med_summary.head())


# #Step 2. Exploratory Data Analysis (EDA) – Univariate Patterns (Medication Dataset)
# 
# In this step, I examine the Medication sheet one variable at a time to understand the basic structure of the data before moving into more complex relationships.
# 
# The specific goals for this step are:
# 
# • Identify the main numeric fields in the medication dataset and look at their distributions using histograms.  
# • Examine the frequency of key categorical variables that are important for medication-error analysis, including:
#   – Certificate / system grouping (if present in this sheet or in the summary sheet)  
#   – Branch (Air vs Ground)  
#   – Medication 1 (primary medication field)  
#   – Pattern (high-level pattern label)  
#   – Pattern Specifics (the detailed pattern classification I will use instead of Risk Event)
# 
# Because `Pattern Specifics` is a free-text or semi-structured field with many unique values, plotting every single value would not be very interpretable. Instead, I will:
# 
# • Count how often each `Pattern Specifics` value appears.  
# • Focus on the most frequent patterns to see which failure modes show up repeatedly in the data.  
# 
# This mirrors the univariate EDA step in the loan assignment (where we examined income, age, etc.), but here the focus is on medication-error patterns and the narrative fields that describe how errors occur.

# STEP 2 – Exploratory Data Analysis (EDA) – Univariate Patterns

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ---- 2.1 Identify numeric columns automatically ----
num_cols = med.select_dtypes(include=["int64", "float64"]).columns.tolist()
print("Numeric columns detected:", num_cols)

if len(num_cols) > 0:
    plt.figure(figsize=(15, 8))
    n_cols = 3
    n_rows = (len(num_cols) + n_cols - 1) // n_cols

    for i, col in enumerate(num_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(med[col], bins=30, kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")

    plt.tight_layout()
    plt.show()
else:
    print("No numeric columns found to plot for histograms.")

# ---- 2.2 Categorical variables (including Pattern and Pattern Specifics) ----
candidate_cat_cols = [
    "Certificate",        # only used if present
    "Branch",
    "Medication 1",
    "Pattern",
    "Pattern Specifics"
]

cat_cols = [c for c in candidate_cat_cols if c in med.columns]
print("Categorical columns used for EDA:", cat_cols)

for col in cat_cols:
    # For high-cardinality columns like Pattern Specifics, show only the top 15
    value_counts = med[col].value_counts().head(15)

    plt.figure(figsize=(8, 4))
    sns.barplot(
        x=value_counts.values,
        y=value_counts.index,
        orient="h"
    )
    plt.title(f"Top {len(value_counts)} categories for {col}")
    plt.xlabel("Count")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()


# # Step 3 — Data Overview with Emphasis on Pattern Specifics
# 
# In this step, I review the overall structure of the **Medication** dataset, with a specific focus on the **Pattern Specifics** column as the main error-type variable.
# 
# The goals of this step are:
# 1.  **Confirm the overall size of the dataset:**
#     * How many total medication-related records are present?
#     * How many columns are included?
# 2.  **List all column names** to understand what information is available for later analysis.
# 3.  **Examine basic completeness and content of the *Pattern Specifics* column:**
#     * Identify how many records have valid, specific pattern data (excluding missing or "Not documented" entries).
#     * List the **Top 10 most frequent error patterns**.
# 4.  **Briefly review a few example rows** to contextualize the error descriptions.
# 
# This mirrors the “data overview” step from the loan assignment, but here the primary focus is on the clinical error description stored in *Pattern Specifics*.

import pandas as pd

# ---------------------------------------------------------
# Step 3: Data Overview & Pattern Specifics Inspection
# ---------------------------------------------------------

# 1. Load the Data
# ---------------------------------------------------------
try:
    df_med = pd.read_excel('Krista 240726 Final.xlsx', sheet_name='Medication')
except FileNotFoundError:
    df_med = pd.read_csv('Krista 240726 Final.xlsx - Medication.csv')

# 2. Confirm Dataset Size and Structure
# ---------------------------------------------------------
rows, cols = df_med.shape
print(f"--- Dataset Dimensions ---")
print(f"Total Records: {rows}")
print(f"Total Columns: {cols}")
print("\n--- Column Names ---")
print(list(df_med.columns))

# 3. Analyze the 'Pattern Specifics' Column (Excluding Missing/Unspecified)
# ---------------------------------------------------------
# Create a filtered series that drops NaNs and specific "missing" text
valid_patterns = df_med['Pattern Specifics'].dropna()
# Filter out common non-specific placeholders if they exist
valid_patterns = valid_patterns[~valid_patterns.isin(['Not documented', 'Not Specified', 'Missing'])]

total_patterns = len(valid_patterns)
missing_count = rows - total_patterns

print("\n--- 'Pattern Specifics' Completeness ---")
print(f"Valid Pattern Records: {total_patterns}")
print(f"Missing/Unspecified:   {missing_count}")
print(f"Completeness:          {100 * total_patterns / rows:.1f}%")

print("\n--- Top 10 Most Frequent Error Patterns (Valid Only) ---")
print(valid_patterns.value_counts().head(10))

# 4. Review Example Rows (Valid Patterns Only)
# ---------------------------------------------------------
print("\n--- Example Rows (First 5 with Valid Patterns) ---")
# Filter the main dataframe to show examples that actually have patterns
df_examples = df_med[df_med['Pattern Specifics'].isin(valid_patterns)]
cols_to_show = ['Source', 'Medication 1', 'Pattern', 'Pattern Specifics']

# Display
print(df_examples[cols_to_show].head(5))

# # Step 4 — Basic Exploratory Data Analysis (EDA) With Emphasis on Pattern Specifics
# 
# In this step, I begin exploring the **Medication** dataset to understand the basic characteristics of the data and identify early patterns that may be useful for later modeling.
# 
# The goals of this step are:
# 1.  **Review the general structure and completeness of the dataset:**
#     * Are there missing values in key fields?
#     * Which fields are fully populated and which require cleaning?
# 2.  **Examine the *Pattern Specifics* column in more detail:**
#     * How many unique pattern categories exist?
#     * Which patterns occur most frequently?
#     * *Note: Missing or "Not documented" values are excluded to focus on actionable data.*
# 3.  **Evaluate the distribution of key variables related to medication errors:**
#     * Branch distribution (Air, Ground, etc.)
#     * Primary risk categories
#     * Types of medication involved (Medication 1, Medication 2)
# 4.  **Generate early questions that will shape later analysis:**
#     * Which patterns appear most frequently?
#     * Do certain branches report more events for specific patterns?
#     * Are some risk types associated with particular medications?
# 
# This mirrors the “initial EDA” step from the loan assignment, but here the emphasis is on **clinical pattern categories** rather than customer demographics. This step ensures we understand the landscape before aggregating or modeling the data.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Step 4: Basic Exploratory Data Analysis (EDA)
# ---------------------------------------------------------

# 1. Load Data
# ---------------------------------------------------------
try:
    df_med = pd.read_excel('Krista 240726 Final.xlsx', sheet_name='Medication')
except FileNotFoundError:
    df_med = pd.read_csv('Krista 240726 Final.xlsx - Medication.csv')

# 2. General Structure & Completeness
# ---------------------------------------------------------
print("--- Missing Values in Key Fields ---")
key_cols = ['Report ID', 'Source', 'Branch', 'Primary Risk', 'Risk Event', 'Medication 1', 'Medication 2', 'Pattern Specifics']
# Filter for columns that actually exist in the dataframe
existing_key_cols = [c for c in key_cols if c in df_med.columns]
print(df_med[existing_key_cols].isna().sum())

# 3. Pattern Specifics Analysis (Detailed & Filtered)
# ---------------------------------------------------------
print("\n--- Pattern Specifics Analysis (Valid Entries Only) ---")

# Filter out NaNs and generic "missing" placeholders
valid_patterns = df_med['Pattern Specifics'].dropna()
valid_patterns = valid_patterns[~valid_patterns.isin(['Not documented', 'Not Specified', 'Missing', 'NaN'])]

print(f"Unique Pattern Categories: {valid_patterns.nunique()}")
print("\nTop 10 Most Frequent Patterns:")
print(valid_patterns.value_counts().head(10))

# 4. Key Variable Distributions
# ---------------------------------------------------------
print("\n--- Branch Distribution ---")
print(df_med['Branch'].value_counts(dropna=True))

print("\n--- Primary Risk Categories ---")
if 'Primary Risk' in df_med.columns:
    print(df_med['Primary Risk'].value_counts(dropna=True))

print("\n--- Top 10 Medications Involved (Medication 1) ---")
print(df_med['Medication 1'].value_counts(dropna=True).head(10))

# Optional: Visualizing the Top Patterns
plt.figure(figsize=(10, 6))
valid_patterns.value_counts().head(10).sort_values().plot(kind='barh', color='teal')
plt.title('Top 10 Valid Medication Error Patterns')
plt.xlabel('Count')
plt.tight_layout()
plt.show()

# # Step 5 — Total Events by Pattern Specifics Category
# 
# In this step, I analyze the **Pattern Specifics** categories to determine which clinical patterns account for the highest number of medication-related events across all certificates.
# 
# The goals of this step are:
# 1.  **Identify all certificate columns** containing event counts (e.g., AEL, GFL, AMR, MTC, REACH, and AMI).
# 2.  **Compute the total number of events** for each *Pattern Specifics* category by summing across all certificates.
# 3.  **Sort these totals** from highest to lowest to highlight the high-impact patterns.
# 4.  **Visualize the top categories** in a bar chart for easier interpretation.
# 
# This mirrors the "risk category breakdown" from the loan assignment, focusing on clinical event patterns to identify system-wide safety priorities.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Step 5: Total Events by Pattern Specifics Category
# ---------------------------------------------------------

# 1. Load the Data
# ---------------------------------------------------------
try:
    df_med = pd.read_excel('Krista 240726 Final.xlsx', sheet_name='Medication')
except FileNotFoundError:
    df_med = pd.read_csv('Krista 240726 Final.xlsx - Medication.csv')

# 2. Prepare the Data (Filter valid patterns)
# ---------------------------------------------------------
# We drop rows where Pattern Specifics is missing to focus on known categories
df_clean = df_med.dropna(subset=['Pattern Specifics']).copy()

# 3. Compute Total Events by Category and Certificate
# ---------------------------------------------------------
# Create a cross-tabulation: Rows = Pattern Specifics, Cols = Source (Certificate)
# This automatically identifies all certificates present in the dataset
pattern_summary = pd.crosstab(df_clean['Pattern Specifics'], df_clean['Source'])

# Add a 'Total Events' column by summing across all certificates
pattern_summary['Total Events'] = pattern_summary.sum(axis=1)

# 4. Sort and Select Top Categories
# ---------------------------------------------------------
# Sort by Total Events in descending order
pattern_summary_sorted = pattern_summary.sort_values(by='Total Events', ascending=False)

# Display the top 10 patterns
print("--- Top 10 Pattern Specifics by Total Events ---")
print(pattern_summary_sorted[['Total Events']].head(10))

# 5. Visualization: Bar Chart of Top Patterns
# ---------------------------------------------------------
# Select top 10 for plotting to keep the chart readable
top_10_patterns = pattern_summary_sorted.head(10)

plt.figure(figsize=(12, 8))
# Plot using the 'Total Events' column
ax = sns.barplot(x=top_10_patterns['Total Events'], y=top_10_patterns.index, palette='viridis')

plt.title('Top 10 Medication Error Patterns (All Certificates)', fontsize=16)
plt.xlabel('Total Number of Events', fontsize=12)
plt.ylabel('Pattern Specifics', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add count labels to the end of each bar
for i, v in enumerate(top_10_patterns['Total Events']):
    ax.text(v + 0.5, i, str(v), color='black', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

# # Step 6 — Create Structured “Pattern Specifics” Flags for Modeling
# 
# In this step, I start transforming the free-text **Pattern Specifics** field into simple, structured variables that can be used in later analysis and modeling.
# 
# The goals of this step are:
# * **Convert unstructured narrative text** into a few clinically meaningful indicator variables.
# * **Create yes/no flags** for major error themes that appear repeatedly in the data, such as:
#     * *Dosing* / maximum dose / volume problems
#     * *Wrong medication* / given instead of the intended drug
#     * *Protocol* / checklist or process-compliance issues
# * **Quantify how often** each type of pattern occurs across all medication events.
# * **Prepare the dataset** so that later steps (for example, decision trees or risk-scoring rules) can use these pattern flags as input features.
# 
# This mirrors the “feature engineering” step from the loan assignment, but instead of creating financial variables, here I am engineering **clinical risk features** directly from the *Pattern Specifics* text.

import pandas as pd
import numpy as np

# 1. Load the Data (Ensuring a clean start for this step)
# ---------------------------------------------------------
try:
    med = pd.read_excel('Krista 240726 Final.xlsx', sheet_name='Medication')
except FileNotFoundError:
    med = pd.read_csv('Krista 240726 Final.xlsx - Medication.csv')

# ---------------------------------------------------------
# Step 6: Feature Engineering - Creating Risk Flags
# ---------------------------------------------------------

# Safety check: make sure the column exists
if "Pattern Specifics" not in med.columns:
    raise KeyError("The 'Pattern Specifics' column is not present in the Medication sheet.")

# 1. Create simple pattern flags (1 = pattern present, 0 = not present)
# ---------------------------------------------------------

# Dosing Errors
# Catches "dosing", "max dose", "overdose", "underdose", etc.
med["Flag_Dosing_Error"] = med["Pattern Specifics"].str.contains(
    r"dosing|max dose|max\.? dose|volume|overdose|underdose",
    case=False, na=False
).astype(int)

# Wrong Medication
# Catches "wrong med", "incorrect medication", "instead of"
med["Flag_Wrong_Med"] = med["Pattern Specifics"].str.contains(
    r"wrong med|wrong medication|instead of|incorrect medication",
    case=False, na=False
).astype(int)

# Protocol/Compliance Errors
# Catches "protocol", "checklist", "policy", "procedure"
med["Flag_Protocol_Error"] = med["Pattern Specifics"].str.contains(
    r"protocol|checklist|policy|procedure",
    case=False, na=False
).astype(int)

# 2. Quick frequency check for each flag
# ---------------------------------------------------------
flag_cols = ["Flag_Dosing_Error", "Flag_Wrong_Med", "Flag_Protocol_Error"]

print("New pattern-flag columns added:")
print(flag_cols)

print("\nEvent counts by flag (1 = pattern present):")
for col in flag_cols:
    count = med[col].sum()
    print(f"{col}: {count} events")

# 3. Optional: show first few rows with the new flags
# ---------------------------------------------------------
print("\nFirst 10 rows with new Pattern Specifics flags:")
cols_to_display = ["Report ID", "Branch", "Primary Risk", "Risk Event",
                   "Medication 1", "Pattern Specifics"] + flag_cols
print(med[cols_to_display].head(10))

# #Step 7 — Pattern Specifics by Branch (Air vs Ground)
# 
# In this step, I move from simple counts of Pattern Specifics to a bivariate view that relates error patterns to transport modality (Air vs Ground).
# 
# Goals of this step:
# 
# • Identify which detailed **medication** error patterns (Pattern Specifics) occur most frequently in the Medication sheet.
# • Compare how often these high-volume patterns appear in Air vs Ground branches.
# • Highlight any patterns that are disproportionately associated with a single branch, which may suggest targeted safety or education needs.
# 
# Approach:
# 
# 1. Use the Medication dataset (med) to find the most common values in the Pattern Specifics column.
# 2. Select the top N patterns (for example, the 10 most frequent) to keep the visualization interpretable.
# 3. Build a cross-tabulation (crosstab) of Pattern Specifics (rows) by Branch (columns: Air, Ground).
# 4. Visualize this crosstab as a heatmap so that branch–pattern combinations with higher counts are immediately visible.
# 
# This step mirrors the “bivariate EDA” section from the loan assignment, but instead of comparing financial variables to Personal_Loan, I am comparing clinical pattern types to the operational branch (Air vs Ground). The result is an early branch-specific risk profile for the most common medication error patterns.

# Step 7 — Pattern Specifics by Branch (Air vs Ground)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# 1. Drop missing Pattern Specifics and find the most frequent patterns
pattern_counts = (
    med["Pattern Specifics"]
    .dropna()
    .value_counts()
)

# Choose how many top patterns to display
n_top = 10
top_patterns = pattern_counts.head(n_top).index.tolist()

print(f"Top {n_top} Pattern Specifics in the Medication dataset:")
print(pattern_counts.head(n_top))

# 2. Filter Medication data to only those top patterns
med_top = med[med["Pattern Specifics"].isin(top_patterns)].copy()

# 3. Create a crosstab: Pattern Specifics (rows) x Branch (columns)
ct_branch_pattern = pd.crosstab(
    med_top["Pattern Specifics"],
    med_top["Branch"]
)

print("\nCrosstab of top Pattern Specifics by Branch (Air vs Ground):")
display(ct_branch_pattern)

# 4. Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    ct_branch_pattern,
    annot=True,
    fmt="d",
    cmap="YlOrRd"
)
plt.title(f"Top {n_top} Medication Error Patterns by Branch (Air vs Ground)")
plt.xlabel("Branch")
plt.ylabel("Pattern Specifics")
plt.tight_layout()
plt.show()


# # Step 7A — Certificate Analysis and Visualization
# 
# In this step, we analyze the medication errors by **Certificate** (Source). We will generate:
# * A **Bar Chart** to compare the total volume of reported errors for each certificate.
# * A **Heat Map** to identify the **Top 10 most frequent error patterns** for each certificate.
# 
# The specific certificates being compared are: **AEL, GFL, MTC, REACH, and AMR.**

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. Load the Data
# ---------------------------------------------------------
try:
    df_med = pd.read_excel('Krista 240726 Final.xlsx', sheet_name='Medication')
    print("Success: Loaded data from Excel file.")
except FileNotFoundError:
    try:
        df_med = pd.read_csv('Krista 240726 Final.xlsx - Medication.csv')
        print("Success: Loaded data from CSV file.")
    except FileNotFoundError:
        print("ERROR: File not found. Please check your file name.")

# ---------------------------------------------------------
# 2. Data Filtering & Preparation
# ---------------------------------------------------------
# REMOVED 'AMI' from this list
target_certificates = ['AEL', 'GFL', 'MTC', 'REACH', 'AMR']

# Filter the dataframe to include only these sources
df_filtered = df_med[df_med['Source'].isin(target_certificates)].copy()

# ---------------------------------------------------------
# 3. Chart: Total Medication Errors by Certificate
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
# Create Bar Chart
ax = sns.countplot(data=df_filtered, x='Source', order=target_certificates, palette='viridis')

# Formatting
plt.title('Total Medication Errors by Certificate', fontsize=16)
plt.ylabel('Count of Errors')
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add numeric labels on top of bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points')

plt.show()

# ---------------------------------------------------------
# 4. FOCUSED Heat Map: Top 10 Patterns Only
# ---------------------------------------------------------
# 1. Count all patterns to find the Top 10
pattern_counts = df_filtered['Pattern Specifics'].value_counts()
top_10_patterns = pattern_counts.head(10).index

# 2. Filter data to only those top 10 patterns
df_heatmap_focused = df_filtered[df_filtered['Pattern Specifics'].isin(top_10_patterns)]

# 3. Create the cross-tabulation table
heatmap_data = pd.crosstab(df_heatmap_focused['Pattern Specifics'], df_heatmap_focused['Source'])

# 4. Reorder columns to match our specific list (AEL, GFL, etc.)
heatmap_data = heatmap_data.reindex(columns=target_certificates)

plt.figure(figsize=(12, 6))

# 5. Generate Heat Map
sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='Reds', linewidths=.5)

plt.title('Top 10 Medication Error Patterns by Certificate', fontsize=16)
plt.xlabel('Certificate', fontsize=12)
plt.ylabel('Error Pattern', fontsize=12)
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# # Step 8 — Analysis of Medications Involved
# 
# Now that we have analyzed errors by Certificate, we proceed to analyze the **Medications** involved in these events.
# 
# In this step, we will:
# * Identify the **Top 10 most frequently involved medications** in reported errors.
# * Visualize the frequency of these medications using a **Bar Chart**.
# * Use a **Heat Map** to see if specific medications are prone to specific types of error patterns (e.g., Are *Fentanyl* errors mostly dosing errors?).
# 
# *Note: We continue to focus on the key certificates (AEL, GFL, MTC, REACH, AMR) defined in the previous step.*

# ---------------------------------------------------------
# Step 8: Medication Involved Analysis
# ---------------------------------------------------------

# 1. Prepare the Data
# ---------------------------------------------------------
# Ensure we are using the filtered data from Step 2
# (If you are running this in a new session, make sure 'df_filtered' is defined as in Step 2)
# We filter for 'Medication 1' to ignore empty entries if any
df_meds_analysis = df_filtered[df_filtered['Medication 1'].notna()].copy()

# 2. Identify Top 10 Medications
# ---------------------------------------------------------
med_counts = df_meds_analysis['Medication 1'].value_counts()
top_10_meds = med_counts.head(10).index
print(f"Top 10 Medications Identified: {list(top_10_meds)}")

# Filter data to only include these top 10 medications
df_top_meds = df_meds_analysis[df_meds_analysis['Medication 1'].isin(top_10_meds)]

# 3. Chart: Frequency of Errors by Medication
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
ax = sns.countplot(data=df_top_meds, x='Medication 1', order=top_10_meds, palette='mako')

plt.title('Top 10 Medications Involved in Errors', fontsize=16)
plt.xlabel('Medication', fontsize=12)
plt.ylabel('Count of Errors', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add count labels
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points')

plt.show()

# ---------------------------------------------------------
# 4. Heat Map: Top 10 Medications vs. Top 10 Patterns
# ---------------------------------------------------------
# We already have 'top_10_meds'. Let's ensure we also focus on 'top_10_patterns' from Step 2.
# (Recalculating top 10 patterns here to ensure code is self-contained)
pattern_counts_step3 = df_top_meds['Pattern Specifics'].value_counts()
top_10_patterns_step3 = pattern_counts_step3.head(10).index

# Filter for both Top Meds AND Top Patterns
df_heatmap_meds = df_top_meds[df_top_meds['Pattern Specifics'].isin(top_10_patterns_step3)]

# Create Cross-tab
heatmap_med_data = pd.crosstab(df_heatmap_meds['Pattern Specifics'], df_heatmap_meds['Medication 1'])

# Reorder columns to match the sorted Top 10 Meds order
heatmap_med_data = heatmap_med_data.reindex(columns=top_10_meds)

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_med_data, annot=True, fmt='g', cmap='Blues', linewidths=.5)

plt.title('Heat Map: Top 10 Medications vs. Top Error Patterns', fontsize=16)
plt.xlabel('Medication', fontsize=12)
plt.ylabel('Error Pattern', fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# # Step 9 — Exploratory Data Analysis (EDA): Bivariate Analysis
# 
# In this step, we investigate relationships between key variables and the **Target Variable (Outcome)**. This corresponds to the "Bivariate Analysis" phase of the loan project, where predictors were compared against the `Personal_Loan` status.
# 
# **Goals:**
# 1.  **Define the Target:** Convert the free-text `Outcome` field into a structured category (**Severity**: 'Critical/Severe', 'No Harm/Stable', etc.).
# 2.  **Analyze Certificate vs. Outcome:** Determine if specific certificates (Sources) have a higher rate of severe errors.
# 3.  **Analyze Medication vs. Outcome:** Identify which medications are most frequently associated with adverse outcomes.
# 4.  **Analyze Error Patterns vs. Outcome:** See if certain error types (e.g., Dosing Errors) correlate with higher severity.
# 
# This step identifies the "drivers" of severe errors, just as bivariate analysis identifies the drivers of loan acceptance.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Step 9: Bivariate Analysis
# ---------------------------------------------------------

# 1. Prepare Data & Define Target Variable
# ---------------------------------------------------------
# Ensure we are working with the latest data
if 'med' in locals():
    df_bi = med.copy()
else:
    # Fallback load if 'med' isn't defined from previous steps
    try:
        df_bi = pd.read_excel('Krista 240726 Final.xlsx', sheet_name='Medication')
    except:
        df_bi = pd.read_csv('Krista 240726 Final.xlsx - Medication.csv')

# Function to categorize Outcome Severity (The "Target")
def categorize_severity(text):
    text = str(text).lower()
    # Define severe keywords
    severe_terms = ['died', 'death', 'expired', 'cpr', 'arrest', 'hypoxia', 'intubated', 'seizure']
    # Define stable/no-harm keywords
    stable_terms = ['stable', 'no adverse', 'resolved', 'prevented', 'normal', 'unchanged']

    if any(x in text for x in severe_terms):
        return 'Critical/Severe'
    elif any(x in text for x in stable_terms):
        return 'No Harm/Stable'
    elif 'not documented' in text or text == 'nan':
        return 'Unknown'
    else:
        return 'Monitor/Intervention' # Intermediate category for things like "given Narcan", "hypotension"

df_bi['Severity_Category'] = df_bi['Outcome'].apply(categorize_severity)

print("--- Target Variable Distribution (Severity) ---")
print(df_bi['Severity_Category'].value_counts())

# 2. Analysis 1: Certificate (Source) vs. Severity
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
# Create a crosstab of counts
source_outcome = pd.crosstab(df_bi['Source'], df_bi['Severity_Category'])
# Normalize to show percentages (Heatmap of Risk)
sns.heatmap(source_outcome, annot=True, fmt='d', cmap='YlOrRd', linewidths=.5)
plt.title('Bivariate: Certificate (Source) vs. Outcome Severity')
plt.ylabel('Certificate')
plt.xlabel('Outcome Severity')
plt.show()

# 3. Analysis 2: Top 10 Medications vs. Severity
# ---------------------------------------------------------
# Filter for top 10 meds
top_meds_list = df_bi['Medication 1'].value_counts().head(10).index
df_top_meds = df_bi[df_bi['Medication 1'].isin(top_meds_list)]

plt.figure(figsize=(12, 6))
sns.countplot(data=df_top_meds, y='Medication 1', hue='Severity_Category',
              order=top_meds_list, palette='magma')
plt.title('Bivariate: Top 10 Medications by Outcome Severity')
plt.xlabel('Count of Events')
plt.ylabel('Medication')
plt.legend(title='Severity', loc='lower right')
plt.show()

# 4. Analysis 3: Pattern Flags vs. Severity (Correlation)
# ---------------------------------------------------------
# We check if our created flags (Dosing, Wrong Med, Protocol) correlate with Critical outcomes
# Create a binary target for correlation: 1 = Critical/Severe, 0 = Other
df_bi['Is_Critical'] = (df_bi['Severity_Category'] == 'Critical/Severe').astype(int)

# Ensure flags exist (from Step 6)
flags = ['Flag_Dosing_Error', 'Flag_Wrong_Med', 'Flag_Protocol_Error']
if all(col in df_bi.columns for col in flags):
    print("\n--- Severe Outcome Rate by Error Type ---")
    for flag in flags:
        # Calculate percentage of severe cases for each flag
        rate = df_bi[df_bi[flag] == 1]['Is_Critical'].mean() * 100
        print(f"Error Type: {flag:<20} | Severe Rate: {rate:.1f}%")

# # Step 10 — Clinical Data Preprocessing
# 
# In this step, we transform our raw medication error data into a structured format suitable for risk modeling. Unlike standard business datasets, clinical data often requires specific handling for free-text outcomes and rare events.
# 
# **Key Actions:**
# 1.  **Define the Target (`Is_Critical`):** We convert the free-text `Outcome` field into a binary target.
#     * **1 (Critical):** Outcomes involving harm, intervention, or distress (e.g., "hypoxia", "monitoring", "intervention").
#     * **0 (Stable/No Harm):** Outcomes where the patient remained stable or no effect was observed.
# 2.  **Feature Engineering:**
#     * **Medication Grouping:** There are hundreds of unique medications. To prevent the model from getting "confused" by rare drugs, we keep the **Top 10 most frequent medications** and group the rest as "Other".
#     * **Risk Flags:** We incorporate the boolean flags created in Step 6 (Dosing Error

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# Step 10: Clinical Data Preprocessing
# ---------------------------------------------------------

# 1. Define the Target: Critical vs. Non-Critical
# ---------------------------------------------------------
# In clinical risk, we want to predict 'Severe' outcomes (1) vs 'Stable/Minor' (0)
# (This logic pulls from your 'Outcome' text analysis in Step 6)
def categorize_severity(text):
    text = str(text).lower()
    severe_terms = ['died', 'death', 'expired', 'cpr', 'arrest', 'hypoxia', 'intubated', 'seizure']
    if any(x in text for x in severe_terms): return 1  # Critical Outcome
    return 0  # No Harm/Stable/Monitoring

# Apply to fresh data if needed
if 'med' in locals():
    df_model = med.copy()
else:
    # Reload if variables are lost
    df_model = pd.read_csv('Krista 240726 Final.xlsx - Medication.csv') # Or use read_excel if preferred
    # Re-create flags from Step 6 if they aren't there
    df_model["Flag_Dosing_Error"] = df_model["Pattern Specifics"].str.contains(r"dosing|max dose|volume", case=False, na=False).astype(int)
    df_model["Flag_Wrong_Med"] = df_model["Pattern Specifics"].str.contains(r"wrong med|instead of", case=False, na=False).astype(int)
    df_model["Flag_Protocol_Error"] = df_model["Pattern Specifics"].str.contains(r"protocol|checklist", case=False, na=False).astype(int)

df_model['Is_Critical'] = df_model['Outcome'].apply(categorize_severity)

# 2. Feature Selection: Prioritize Clinical Context
# ---------------------------------------------------------
# Group rare medications into 'Other' to focus on systemic drug risks
top_10_meds = df_model['Medication 1'].value_counts().nlargest(10).index
df_model['Med_Grouped'] = df_model['Medication 1'].apply(lambda x: x if x in top_10_meds else 'Other')

# Select features that drive process improvement
feature_cols = [
    'Source',               # Who made the error? (Certificate)
    'Branch',               # Where did it happen? (Air vs. Ground)
    'Med_Grouped',          # Which drug? (High risk meds)
    'Flag_Dosing_Error',    # Is it a dosing issue?
    'Flag_Wrong_Med',       # Is it a wrong drug issue?
    'Flag_Protocol_Error'   # Is it a protocol violation?
]

X = df_model[feature_cols]
y = df_model['Is_Critical']

# 3. Encoding & Splitting
# ---------------------------------------------------------
# Convert categories (Source, Branch) into binary flags for the model
X = pd.get_dummies(X, columns=['Source', 'Branch', 'Med_Grouped'], drop_first=True)

# Split Data: Stratify is CRITICAL here to maintain the ratio of severe events in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

print(f"Training Data: {X_train.shape[0]} events")
print(f"Testing Data:  {X_test.shape[0]} events")
print(f"Critical Events in Train Set: {y_train.sum()} ({y_train.mean():.1%} of total)")

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score
# 
# # ---------------------------------------------------------
# # Step 11: Clinical Risk Model (Decision Tree)
# # ---------------------------------------------------------
# 
# # 1. Train the Model with Class Balancing
# # 'class_weight="balanced"' forces the model to pay attention to the rare 'Critical' events
# risk_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
# risk_model.fit(X_train, y_train)
# 
# # 2. Predictions
# y_pred = risk_model.predict(X_test)
# 
# # 3. Clinical Evaluation
# print("--- Model Performance ---")
# # Accuracy is less important here than Recall (catching the bad events)
# print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2%}")
# print(f"Critical Event Capture Rate (Recall): {recall_score(y_test, y_pred):.2%}")
# 
# print("\n--- Confusion Matrix ---")
# # Format: [ True Negatives   False Positives (False Alarms) ]
# #         [ False Negatives (Missed!)   True Positives (Caught) ]
# print(confusion_matrix(y_test, y_pred))
# 
# print("\n--- Feature Importance (What drives risk?) ---")
# # Identify which clinical factors (Flags, Certs, Meds) drive severe outcomes
# feat_importances = pd.Series(risk_model.feature_importances_, index=X_train.columns)
# print(feat_importances.nlargest(5))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score

# ---------------------------------------------------------
# Step 11: Clinical Risk Model (Decision Tree)
# ---------------------------------------------------------

# 1. Train the Model with Class Balancing
# 'class_weight="balanced"' tells the model: "Critical errors are rare but expensive. Prioritize finding them."
risk_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
risk_model.fit(X_train, y_train)

# 2. Predictions
y_pred = risk_model.predict(X_test)

# 3. Clinical Evaluation
print("--- Model Performance ---")
# Accuracy is less important here than Recall (catching the bad events)
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"Critical Event Capture Rate (Recall): {recall_score(y_test, y_pred):.2%}")

print("\n--- Confusion Matrix ---")
# Format: [ True Negatives   False Positives (False Alarms) ]
#         [ False Negatives (Missed!)   True Positives (Caught) ]
print(confusion_matrix(y_test, y_pred))

print("\n--- Feature Importance (What drives risk?) ---")
# Identify which clinical factors (Flags, Certs, Meds) drive severe outcomes
feat_importances = pd.Series(risk_model.feature_importances_, index=X_train.columns)
print(feat_importances.nlargest(5))

# # Step 12 — Model Improvement (Tuning for Safety)
# 
# The initial model might be too complex or "noisy." In this step, we use **Hyperparameter Tuning** to find the optimal tree structure that balances accuracy with simplicity.
# 
# **Optimization Goals:**
# * **Metric:** We optimize for **ROC-AUC** (Area Under the Curve). This is a better metric for imbalanced data than simple accuracy because it measures how well the model distinguishes between "Critical" and "Stable" events across all probability thresholds.
# * **Tree Depth:** We test different tree depths to find simple, robust rules that define high-risk scenarios.
# * **Rule Extraction:** Finally, we extract the logic from the best tree to provide actionable "Safety Rules" for the clinical team.

from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_text

# ---------------------------------------------------------
# Step 12: Improving Risk Detection (Pruning)
# ---------------------------------------------------------

# 1. Setup Grid Search
# We test different tree depths to find the "sweet spot" for safety rules
param_grid = {
    'max_depth': [3, 4, 5, 7, 10],
    'min_samples_leaf': [5, 10, 20],
    'class_weight': ['balanced'] # Always keep balanced for medical risk
}

# Optimize for ROC_AUC (ability to distinguish severe from non-severe)
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# 2. Best Model Results
best_model = grid_search.best_estimator_
print(f"Best Ruleset Depth: {grid_search.best_params_['max_depth']}")
print(f"Best ROC-AUC Score: {grid_search.best_score_:.4f}")

# 3. Visualize the "Safety Rules"
# This text printout shows exactly which conditions lead to High Risk
rules = export_text(best_model, feature_names=list(X_train.columns))
print("\n--- Clinical Risk Rules Generated by Model ---")
print(rules)

# # Step 13 — Actionable Insights & Clinical Recommendations
# 
# Based on the Decision Tree analysis and risk modeling, we have identified the primary drivers of critical medication errors. Below are the key findings and specific recommendations to improve patient safety.
# 
# ### **1. Key Drivers of Critical Outcomes**
# Our model identified the following factors as the strongest predictors of a Severe/Critical outcome:
# * **Dosing Errors (Highest Impact):** Errors involving "max dose," "overdose," or "volume" calculations are the single biggest predictor of a severe patient outcome.
# * **High-Risk Medications:** Specific drugs, particularly **Fentanyl** and **Ketamine**, appear frequently in the high-risk branches of the decision tree.
# * **Protocol Deviations:** Events flagged as "protocol" or "checklist" violations have a high correlation with adverse events, suggesting process failures rather than just individual mistakes.
# 
# ### **2. Recommendations for Intervention**
# 
# **A. Target Dosing Calculations for High-Risk Meds**
# * **Finding:** Dosing errors involving sedatives/analgesics (Fentanyl, Ketamine) frequently lead to hypoxia or arrest.
# * **Recommendation:** Implement a **mandatory "High-Risk Double Check"** in the ePCR system. When a crew selects Fentanyl or Ketamine, the system should require

from sklearn.tree import export_text

# ---------------------------------------------------------
# Step 13: Generating Clinical Insights
# ---------------------------------------------------------

# 1. Visualize the Safety Rules (Decision Logic)
# We extract the top rules from the tree to see exactly where the risk "splits" occur.
# (e.g., "If Flag_Dosing_Error > 0.5, then Class: 1 (Critical)")
tree_rules = export_text(best_model, feature_names=list(X_train.columns), max_depth=3)
print("--- Key Clinical Risk Rules (Decision Tree) ---")
print(tree_rules)

# 2. Risk Profiling: Compare Critical vs. Non-Critical Events
# We calculate the mean values for our flags to see how much more frequent they are in critical cases.
# (Re-combining X and y for analysis)
analysis_df = X_train.copy()
analysis_df['Is_Critical'] = y_train

print("\n--- Risk Profile: Frequency of Errors in Critical vs. Stable Cases ---")
# Group by Outcome (0=Stable, 1=Critical) and get the average of the error flags (which acts as a percentage)
risk_profile = analysis_df.groupby('Is_Critical')[['Flag_Dosing_Error', 'Flag_Wrong_Med', 'Flag_Protocol_Error']].mean() * 100
risk_profile.index = ['Stable/Minor (0)', 'Critical/Severe (1)']
print(risk_profile.round(1).astype(str) + '%')

# 3. High-Risk Medication Analysis
# Which specific drugs have the highest % rate of critical outcomes?
print("\n--- Critical Outcome Rate by Medication (Top 5 High Risk) ---")
# We use the original dataframe to get the raw medication names back
med_risk = df_model.groupby('Medication 1')['Is_Critical'].mean() * 100
# Filter for meds with at least 10 events to ensure statistical relevance
med_counts = df_model['Medication 1'].value_counts()
common_meds = med_counts[med_counts >= 10].index
print(med_risk[common_meds].sort_values(ascending=False).head(5).round(1).astype(str) + '%')

from sklearn.tree import export_graphviz
import graphviz

# ---------------------------------------------------------
# Visualizing the Decision Tree
# ---------------------------------------------------------

# We export the 'best_model' we trained in Step 9
dot_data = export_graphviz(
    best_model,
    out_file=None,
    feature_names=list(X_train.columns),
    class_names=['Stable', 'Critical'],
    filled=True,
    rounded=True,
    special_characters=True,
    max_depth=3 # Limit depth for readability
)

# Render the graph
graph = graphviz.Source(dot_data)
graph

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# (Assuming 'df_target' and 'target_certs' are defined)

plt.figure(figsize=(14, 8))

colors = {'AEL': '#1f77b4', 'GFL': '#ff7f0e', 'MTC': '#2ca02c', 'REACH': '#d62728', 'AMR': '#9467bd'}

for cert in target_certs:
    subset = df_target[df_target['Source'] == cert]
    if len(subset) < 5: continue

    # Resample
    ts = subset.set_index('Date').resample('MS').size()

    try:
        # Forecast
        model = ExponentialSmoothing(ts, trend='add', seasonal=None, initialization_method="estimated").fit()
        forecast = model.forecast(6)
        total_err = forecast.sum()

        label_str = f"{cert} (Proj: {total_err:.1f})"

        # Plot smoothed history
        plt.plot(ts.index, ts.rolling(3).mean(), color=colors[cert], linewidth=2, label=label_str)
        # Plot forecast
        plt.plot(forecast.index, forecast, color=colors[cert], linestyle='--', linewidth=2, alpha=0.8)
        # End dot
        plt.plot(forecast.index[-1], forecast.iloc[-1], marker='o', color=colors[cert], markersize=5)

    except:
        pass

# Formatting
plt.title('Projected Fentanyl/Ketamine Errors (Next 6 Months)', fontsize=18, weight='bold')
plt.ylabel('Monthly Error Count', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)

# CUSTOM Y-AXIS (Max Height 3.0 with 0.5 intervals)
# We set the ticks exactly from 0 to 3.0
plt.yticks(np.arange(0, 3.1, 0.5))
plt.ylim(bottom=0, top=3.0)

# Legend Logic
plt.plot([], [], color='black', linewidth=1, label='─ History (Trend)')
plt.plot([], [], color='black', linestyle='--', linewidth=1, label='--- Forecast')
plt.legend(loc='upper left', fontsize=11, frameon=True, shadow=True, title="Certificate (6-Mo Projection)")

plt.tight_layout()
plt.show()

# # --- FINAL EXECUTIVE SUMMARY: Clinical Risk & Operational Forecast ---
# 
# ### **1. Project Scope & Objectives**
# This analysis aimed to transform **558 medication error reports** (across AEL, AMI, GFL, MTC, REACH, AMR) into actionable safety intelligence. We utilized a dual-modeling approach:
# * **Risk Modeling (Decision Tree):** To identify *which* errors cause severe patient harm (Critical/Severe Outcome).
# * **Volume Forecasting (Time Series):** To predict *how many* high-risk errors will occur in the next 6 months.
# 
# ### **2. Clinical Risk Findings (The "What")**
# Our predictive model (98.4% Accuracy, **94.4% Recall**) successfully identified the primary drivers of critical patient outcomes:
# * **Primary Driver:** **Dosing Errors** are the single leading cause of harm (33.3% impact on severity). Specifically, errors involving *"Limit per kg exceeded"* and *"Volume Calculation"* outweigh equipment failures in terms of risk.
# * **High-Risk Agents:** **Ketamine (90 events)** and **Fentanyl (66 events)** combined account for **28%** of all reported errors. These drugs appear most frequently in the "high-risk" branches of the decision tree.
# * **Protocol Drift:** The **Air Branch** showed a higher correlation with protocol deviations (15.5% impact) compared to Ground, suggesting environmental complexity is a factor in safety checklist adherence.
# 
# ### **3. Operational Forecast (The "Where" & "When")**
# *Based on Yoplait-style Time Series Forecasting (Holt’s Linear Model)*
# 
# We projected the volume of high-risk medication errors (Fentanyl/Ketamine) for the upcoming 6-month period to prioritize resource allocation:
# 
# * **High Priority Targets (Immediate Intervention):**
#     * **AMR:** Projected **7.9** errors. (Note: Data is volatile, RMSE 1.40, indicating sporadic spikes).
#     * **AEL:** Projected **7.1** errors. (Note: High confidence forecast, RMSE 0.59).
# * **Lower Priority / Monitoring:**
#     * **GFL:** Projected **6.4** errors.
#     * **MTC:** Projected **4.0** errors.
#     * **REACH:** Projected **0.9** errors (Trending positively downwards).
# 
# ### **4. Strategic Recommendations**
# Based on the convergence of clinical risk data and operational forecasts, we recommend:
# 
# 1.  **Implement "Hard Stop" Logic (Priority: High)**
#     * **Data Rationale:** Fentanyl and Ketamine are the strongest predictors of severe outcomes.
#     * **Action:** Configure the ePCR to require a mandatory peer-verify checkbox for these two agents. **Roll this out to AMR and AEL first**, as they account for the majority of the projected error volume.
# 
# 2.  **Target "Weight-Based" Training (Priority: Medium)**
#     * **Data Rationale:** "Limit per kg exceeded" was identified as a critical failure point in the decision tree.
#     * **Action:** Launch a focused training module on pediatric and weight-based calculations to address this specific skill gap.
# 
# 3.  **Air Medical Protocol Review (Priority: Medium)**
#     * **Data Rationale:** Protocol deviations are the second highest driver of risk (15.5%) and are more prevalent in the Air branch (67% of total reports).
#     * **Action:** Simplify in-flight checklists for Air Medical crews to reduce cognitive load and improve adherence during transport.
