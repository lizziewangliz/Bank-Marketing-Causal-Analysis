# Bank Marketing Campaign: Causal Impact of Previous Contact on Customer Subscription

## Project Summary

This project investigates whether customers previously contacted by a bank are more likely to subscribe to a term deposit. Using observational data from a Portuguese bank’s marketing campaigns, we apply both econometric and machine learning-based causal inference methods to estimate the true effect of prior contact on subscription likelihood.

Our objective is to inform more efficient customer outreach strategies by identifying which segments respond positively to follow-up contact—and which may not.

---

## Research Question

**Does prior customer contact increase the likelihood of subscribing to a term deposit, and for whom?**

---

## Dataset Overview

- **Source**: Public dataset from Moro et al. (2011), originally used for modeling direct marketing effectiveness.
- **Observations**: 4,521 customers
- **Features**: Demographics, financial status, contact history, marketing outcomes

**Key variables**:
- `previous`: number of past contacts (treatment)
- `y`: whether customer subscribed to a term deposit (outcome)
- Control variables: age, job, marital status, education, balance, loan status, contact type, etc.

---

## Methods Used

We combined traditional causal inference with modern machine learning approaches:

1. **OLS Regression** – Baseline correlation estimate
2. **Propensity Score Matching (PSM)** – Controls for selection bias using matched samples
3. **Regression Adjustment** – Estimates treatment effect conditional on covariates
4. **Difference-in-Differences (DiD)** – Simulates a before-and-after comparison using contact intensity
5. **Meta Learners (T-Learner)** – Estimates individual-level treatment effects using Random Forests
6. **Causal Random Forest (CRF)** – Captures heterogeneous treatment effects and identifies key drivers

---

## Key Findings

- Prior contact **significantly increases** the likelihood of subscription (~11 percentage points on average).
- **Call duration** is the strongest predictor of conversion across all models.
- **Repeated outreach** shows diminishing or even negative returns (DiD suggests potential fatigue).
- Meta Learner and CRF analyses reveal:
  - **High-balance, well-educated, married customers** benefit most from follow-up.
  - **Blue-collar, retired, and self-employed** segments are less responsive.
  - Feature importance varies by model, but **engagement quality** matters more than frequency.

---

## Files Included

- `Bank Marketing.R`: Full R script with all steps—from data cleaning to advanced modeling
- `bank.csv`: Original dataset (semicolon-delimited)
- `bank_cleaned_original.csv`: Cleaned version used for modeling
- `Final Project Presentation.pdf`: Summary deck with key insights, methods, and plots
- `README.txt`: Legacy readme with technical notes and variable definitions

---

## How to Run

1. Clone or download this repository.
2. Open `Bank Marketing.R` in RStudio.
3. Ensure required packages are installed (see below).
4. Run the script section by section or all at once.
5. Review outputs (model summaries, plots, treatment effect graphs).

**Required R packages**:
```r
install.packages(c("dplyr", "ggplot2", "MatchIt", "grf", "caret", "lmtest", "sandwich", "corrplot", "stargazer"))
