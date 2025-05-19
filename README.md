# Bank Marketing Analysis


## **1. Files Included**
### **R Code File**
- **Bank Marketing.R** The main R script containing the full analysis, including data preparation, exploratory data analysis (EDA), and causal inference methods (OLS, PSM, Regression Adjustment, DiD, Meta Learners, and Causal Random Forests).

### **Datasets**
- **bank.csv** The original dataset, containing 4,521 observations from the publicly available Bank Marketing dataset. This dataset includes customer demographics, past interactions, and marketing outcomes.
- **bank_cleaned_original.csv** The cleaned dataset used in the R code after preprocessing, including necessary transformations such as categorical encoding, missing value handling, and variable adjustments.

---

## **2. Dataset Information**
### **Source**  
The dataset originates from the following study:
- **[Moro et al., 2011]** S. Moro, R. Laureano, and P. Cortez. *Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology.* In *Proceedings of the European Simulation and Modelling Conference (ESM'2011),* pp. 117-121, Guimarães, Portugal, October 2011. EUROSIS.
- **Available at:**
  - PDF: [http://hdl.handle.net/1822/14838](http://hdl.handle.net/1822/14838)
  - BIB: [http://www3.dsi.uminho.pt/pcortez/bib/2011-esm-1.txt](http://www3.dsi.uminho.pt/pcortez/bib/2011-esm-1.txt)

---

Input variables:
   # bank client data:
   1 - age (numeric)
   2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services") 
   3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
   4 - education (categorical: "unknown","secondary","primary","tertiary")
   5 - default: has credit in default? (binary: "yes","no")
   6 - balance: average yearly balance, in euros (numeric) 
   7 - housing: has housing loan? (binary: "yes","no")
   8 - loan: has personal loan? (binary: "yes","no")
   # related with the last contact of the current campaign:
   9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
  10 - day: last contact day of the month (numeric)
  11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
  12 - duration: last contact duration, in seconds (numeric)
   # other attributes:
  13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
  14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
  15 - previous: number of contacts performed before this campaign and for this client (numeric)
  16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

  Output variable (desired target):
  17 - y - has the client subscribed a term deposit? (binary: "yes","no")

---

## **4. Analysis Workflow**
The analysis follows these main steps:

### **Data Preparation**
- Import the `bank.csv` dataset.
- Clean and preprocess data, handling categorical variables and missing values.
- Save the cleaned dataset as `bank_cleaned_original.csv`.

### **Exploratory Data Analysis (EDA)**
- Compute summary statistics for key variables.
- Visualize distributions and correlations between previous contact, call duration, and subscription likelihood.

### **Causal Inference Methods**
Several techniques are applied to estimate the causal impact of previous contact on subscription rates:

- **OLS Regression (Baseline Model)** ? Provides an initial estimate of the effect of previous contact.
- **Propensity Score Matching (PSM)** ? Matches similar customers in treatment and control groups to reduce selection bias.
- **Regression Adjustment** ? Controls for customer demographics and financial variables.
- **Difference-in-Differences (DiD)** ? Explores changes in subscription rates over time between contacted and non-contacted customers.
- **Meta Learners (T-Learner Approach)** ? Uses machine learning to estimate heterogeneous treatment effects.
- **Causal Random Forests (CRF)** ? Identifies which customer characteristics drive treatment effects.

### **Interpretation of Results**
- Evaluate the statistical significance of results.
- Identify customer segments most likely to benefit from previous contact.
- Compare findings to prior research.

---

## **5. Instructions for Running the Code**
1. **Ensure all files are in the same directory.**
2. **Open RStudio** (or any R environment).
3. **Run `Bank Marketing.R`** step by step or execute the full script.
4. **Review output files** (graphs, tables, model results).
5. **Modify parameters if needed** (e.g., adjust matching ratios in PSM, change regression models, explore alternative machine learning methods).

---

## **6. Requirements**
To successfully run the `Bank Marketing.R` script, ensure the following R packages are installed:

install.packages(c("tidyverse", "MatchIt", "grf", "rdrobust", "lmtest", "sandwich", "ggplot2", "broom"))
