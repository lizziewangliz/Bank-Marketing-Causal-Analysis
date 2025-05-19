### Empirical Economic Final Project ###
### Team 32 ###
### Does Previous Contact Influence Subscription Rates? ###
### We are exploring whether customers who were previously contacted are more likely to subscribe to a term deposit ###

########################
### Data Preparation ###
########################

# Load necessary libraries
library(dplyr)
library(readr)

# Load the dataset with the correct delimiter
df <- read.csv("bank.csv", sep = ";", header = TRUE, stringsAsFactors = FALSE)

# Convert binary categorical variables to numeric (yes=1, no=0)
df <- df %>%
  mutate(
    default = ifelse(default == "yes", 1, 0),
    housing = ifelse(housing == "yes", 1, 0),
    loan = ifelse(loan == "yes", 1, 0),
    y = ifelse(y == "yes", 1, 0)  # Target variable
  )

# Convert categorical variables to factors (but keep them in original structure)
df <- df %>%
  mutate(
    job = as.factor(job),
    marital = as.factor(marital),
    education = as.factor(education),
    contact = as.factor(contact),
    month = as.factor(month),
    poutcome = as.factor(poutcome)
  )

# Handle missing values (if any exist)
df <- na.omit(df)

# Save the cleaned dataset while **keeping the original structure**
write.csv(df, "bank_cleaned_original.csv", row.names = FALSE)

# Display a summary of the cleaned dataset
summary(df)

# Print column names to ensure structure is maintained
print(colnames(df))


#######################################
### Exploratory Data Analysis (EDA) ###
#######################################

# Load necessary libraries
library(dplyr)
library(ggplot2)
install.packages("corrplot")
library(corrplot)

# Display first few rows
print(head(df))

# Display structure and summary statistics
print(str(df))
print(summary(df))

# Count target variable distribution (Subscription rate)
subscription_count <- table(df$y)
print(subscription_count)
barplot(subscription_count, col = c("red", "green"), main = "Subscription Distribution (y)",
        names.arg = c("No", "Yes"))

# Visualizing Numeric Features
# Histogram for age
ggplot(df, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "blue", color = "black", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Age Distribution", x = "Age", y = "Count")

# Histogram for balance
ggplot(df, aes(x = balance)) +
  geom_histogram(binwidth = 500, fill = "purple", color = "black", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Balance Distribution", x = "Balance", y = "Count")

# Boxplot for duration (Call Duration)
ggplot(df, aes(y = duration)) +
  geom_boxplot(fill = "orange", color = "black", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Boxplot of Call Duration", y = "Call Duration (seconds)")

# Categorical Variable Analysis
# Bar plot for job distribution
ggplot(df, aes(x = job)) +
  geom_bar(fill = "lightblue", color = "black") +
  theme_minimal() +
  labs(title = "Job Distribution", x = "Job", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Correlation Analysis (Only numeric variables)
numeric_cols <- df %>% select_if(is.numeric)
corr_matrix <- cor(numeric_cols)

# Correlation heatmap
corrplot(corr_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)

# Save correlation plot as an image
png("correlation_plot.png")
corrplot(corr_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)
dev.off()

# Checking missing values
missing_values <- colSums(is.na(df))
print(missing_values)


##########################
### OLS Baseline Model ###
##########################

# Load necessary libraries
library(dplyr)
library(stats)

# Ensure the dependent variable (y) is numeric for OLS regression
df$y <- as.numeric(df$y)  # Convert 'y' back to numeric (0/1)

### OLS Model: Effect of Previous Marketing Contact on Subscription
# Treatment: Clients who were contacted before (`previous > 0`)
# Control: Clients who were never contacted before (`previous = 0`)
model_previous <- lm(y ~ previous + age + balance + duration + job + marital + education + housing + loan, data = df)
summary(model_previous)


###################################
### Matching (Propensity Score) ###
###################################

# Load necessary libraries
library(dplyr)
library(MatchIt)
library(ggplot2)

# Define treatment and control groups
df <- df %>%
  mutate(treated = ifelse(previous > 0, 1, 0))  # 1 = contacted before, 0 = never contacted

# Define covariates for matching (excluding 'previous' to prevent bias)
covariates <- c("age", "balance", "duration", "campaign", "housing", "loan", "job", "marital", "education", "contact")

# Estimate propensity score using logistic regression
ps_model <- glm(treated ~ age + balance + duration + campaign + housing + loan + job + marital + education + contact, 
                family = binomial(), data = df)

# Perform Propensity Score Matching (Nearest Neighbor, 1:1 Matching)
matched_data <- matchit(treated ~ age + balance + duration + campaign + housing + loan + job + marital + education + contact, 
                        data = df, method = "nearest", ratio = 1)

# Summary of matching
summary(matched_data)

# Create a dataframe with matched observations
df_matched <- match.data(matched_data)

# Check balance before and after matching
plot(matched_data, type = "hist")  # Histogram of propensity scores
plot(matched_data, type = "qq")    # Q-Q plot of covariate balance

# Estimate treatment effect (ATE & ATT) using OLS on matched data
model_psm <- lm(y ~ treated, data = df_matched)
summary(model_psm)

# Convert treated variable to a factor for categorical plotting
df_matched$treated <- as.factor(df_matched$treated)

# Calculate mean subscription rate for each group
subscription_summary <- df_matched %>%
  group_by(treated) %>%
  summarise(mean_subscription = mean(y))

# Create a bar plot
ggplot(subscription_summary, aes(x = treated, y = mean_subscription, fill = treated)) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_manual(values = c("red", "green")) +
  theme_minimal() +
  labs(title = "Mean Subscription Rate by Previous Contact",
       x = "Previous Contact (1 = Yes, 0 = No)",
       y = "Mean Subscription Rate") +
  theme(legend.position = "none")  # Remove unnecessary legend

# The results from Propensity Score Matching (PSM) suggest that previous contact 
# significantly increases the likelihood of a customer subscribing to a term deposit.

# The estimated treatment effect (coefficient of 'treated') is 0.11152, 
# meaning that customers who were previously contacted are, on average, 
# 11.15 percentage points more likely to subscribe compared to those who were never contacted.

# The effect is highly statistically significant (p-value = 1.65e-09), 
# indicating strong evidence that previous marketing contact impacts subscription behavior.

# After matching, we achieved a well-balanced dataset with 816 treated and 816 control units, 
# ensuring that the treatment and control groups are comparable.

# However, the Adjusted R-squared value (0.02146) is low, meaning that while previous contact 
# has a significant effect, other factors likely influence the subscription decision as well.

# Overall, this analysis supports the causal claim that previous contact increases 
# the likelihood of subscription, but further refinements (such as including additional confounders) 
# could help improve the explanatory power of the model.


#############################
### Regression Adjustment ###
#############################

# Load necessary libraries
library(stargazer)  

# Define the treatment variable (previous contact)
df <- df %>%
  mutate(treated = ifelse(previous > 0, 1, 0))  # 1 = contacted before, 0 = never contacted

# Fit the OLS Regression Model with Controls
model_ra <- lm(y ~ treated + age + balance + duration + campaign + housing + loan + job + marital + education + contact, 
               data = df)

# Display regression results
summary(model_ra)

# Export the regression table for reporting
stargazer(model_ra, type = "text", title = "Regression Adjustment Results", align = TRUE)

# Visualizing the Effect of Treatment
ggplot(df, aes(x = as.factor(treated), y = y)) +
  geom_bar(stat = "summary", fun = "mean", fill = c("red", "green")) +
  theme_minimal() +
  labs(title = "Effect of Previous Contact on Subscription",
       x = "Previous Contact (1 = Yes, 0 = No)",
       y = "Mean Subscription Rate")

# Our Regression Adjustment results reinforce what we found in PSM:  
# Previous contact significantly increases the likelihood of subscription.  
# Customers who were previously contacted are 10.9 percentage points more likely to subscribe (p < 0.001).  
# This closely aligns with our PSM estimate (11.15 percentage points), confirming the robustness of our findings.

# Key Insights:  
# Call duration remains a strong predictor of subscription.  
# Customers with housing or personal loans are less likely to subscribe (p < 0.001).  
# Retired (+), Students (+), Blue-collar (-), and Unemployed (-) groups show significant effects.  
# Unknown contact methods negatively impact subscription rates.  

# Model Takeaways:
# The model explains 21.5% of variation in subscription (Adjusted R² = 0.215).  
# The treatment effect is highly significant (p < 0.001), further validating our causal claim.  

# Conclusion & Next Steps:  
# The consistency between Regression Adjustment and PSM strengthens our confidence in the results.  


################################
### Difference-in-Difference ###
################################

# Load necessary libraries
library(dplyr)
library(ggplot2)
library(lmtest)
library(sandwich)

# Define treatment and control groups
df <- df %>%
  mutate(treated = ifelse(previous > 0, 1, 0))  # 1 = contacted before, 0 = never contacted

# Define "time" variable using `campaign`
df <- df %>%
  mutate(time = ifelse(campaign == 1, 0, 1))  # 0 = single contact (pre-period), 1 = multiple contacts (post-period)

# Create interaction term for Difference-in-Differences (DiD)
df <- df %>%
  mutate(treated_time = treated * time)

# Estimate DiD model
did_model <- lm(y ~ treated + time + treated_time + age + balance + duration + housing + loan + job + marital + education + contact, 
                data = df)

# Display results
summary(did_model)

# Clustered standard errors (optional for robustness)
coeftest(did_model, vcov = vcovHC(did_model, type = "HC1"))

# **Visualizing the DiD Effect**
df_grouped <- df %>%
  group_by(treated, time) %>%
  summarise(mean_subscription = mean(y), .groups = 'drop')

ggplot(df_grouped, aes(x = as.factor(time), y = mean_subscription, group = as.factor(treated), color = as.factor(treated))) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  labs(title = "Difference-in-Differences: Effect of Previous Contact on Subscription (Using Campaign)",
       x = "Campaign as Time Variable (0 = Single Contact, 1 = Multiple Contacts)",
       y = "Mean Subscription Rate",
       color = "Treatment Group") +
  theme_minimal()

# Our DiD analysis finds that previous contact significantly increases subscription rates.  
# Customers who were previously contacted (`treated`) are 11.12 percentage points more likely to subscribe (p < 0.001).  
# However, the interaction term (`treated_time`) is not significant (p = 0.645), meaning the effect of multiple contacts  
# compared to a single contact does not show a clear additional boost in subscription rates.  

# Key Takeaways: 
# Time (`campaign > 1`) alone has a negative impact on subscriptions (-2.55 percentage points, p = 0.009).  
# Call duration remains a strong predictor of subscription (p < 0.001).  
# Housing loans (-), personal loans (-), and certain job categories (blue-collar, unemployed) show significant negative effects.  
# Retired customers (+) and married individuals (-) have significant impacts on subscription likelihood.  
# Unknown contact methods significantly reduce subscription rates (-4.73 percentage points, p < 0.001). 

# Final Conclusion:
# The lack of significance in `treated_time` suggests that additional contacts do not substantially enhance subscription rates  
# beyond the first one. This aligns with the idea that the first contact is the most impactful in driving customer decisions.  
# Our results remain consistent with findings from PSM and Regression Adjustment, reinforcing the causal impact of previous contact.  
# We may want to explore Fixed Effects or Causal Machine Learning to further refine our approach.  


#####################
### Meta Learners ###
#####################


# Load necessary libraries
library(dplyr)
library(caret)
install.packages("grf")
library(grf)  # For Causal Forests

# Load dataset
df <- read.csv("bank_cleaned_original.csv", header = TRUE, stringsAsFactors = TRUE)

# Define treatment variable (previous contact)
df <- df %>%
  mutate(treated = ifelse(previous > 0, 1, 0))  # 1 = contacted before, 0 = never contacted

# Split data into treated and control groups
df_treated <- df %>% filter(treated == 1)
df_control <- df %>% filter(treated == 0)

# Define features (X) and outcome (y)
features <- c("age", "balance", "duration", "campaign", "housing", "loan", "job", "marital", "education", "contact")
X_treated <- df_treated[, features]
X_control <- df_control[, features]
y_treated <- df_treated$y
y_control <- df_control$y

# Train two separate models for T-Learner
model_treated <- train(X_treated, y_treated, method = "rf", trControl = trainControl(method = "cv", number = 5))  # Random Forest
model_control <- train(X_control, y_control, method = "rf", trControl = trainControl(method = "cv", number = 5))

# Estimate Individual Treatment Effects (ITE)
X_all <- df[, features]  # Features for all customers
pred_treated <- predict(model_treated, X_all)
pred_control <- predict(model_control, X_all)

# Compute ITE for each customer
df$ITE <- pred_treated - pred_control

# Visualizing the Distribution of Treatment Effects
ggplot(df, aes(x = ITE)) +
  geom_histogram(bins = 30, fill = "blue", alpha = 0.6) +
  theme_minimal() +
  labs(title = "Distribution of Estimated Individual Treatment Effects (ITE)",
       x = "Individual Treatment Effect (ITE)",
       y = "Count")

# Summarizing Treatment Effects
summary(df$ITE)

# Our analysis using Meta Learners (T-Learner) reveals that previous contact has a positive causal effect on subscription,  
# but the impact varies significantly across customers.  

# Key Findings:  
# On average, previous contact increases subscription probability by ~11.87%.  
# Most customers benefit (~75% have a positive ITE), but some experience a negative effect.
# Highest observed effect: +68% increase in subscription probability.  
# Lowest observed effect: -55% decrease in subscription probability.  

# Not all customers respond positively to contact some are less likely to subscribe after being contacted. 
# This suggests a need for personalized targeting to avoid wasting marketing efforts on low-response customers.  
# Next, we will analyze which customer characteristics (age, balance, job, etc.) are associated with high or low treatment effects.  

# Load necessary libraries
library(dplyr)
library(ggplot2)

# Define high and low ITE groups
df <- df %>%
  mutate(ITE_group = case_when(
    ITE >= quantile(ITE, 0.75) ~ "High ITE",
    ITE <= quantile(ITE, 0.25) ~ "Low ITE",
    TRUE ~ "Medium ITE"
  ))

#  Summary Statistics: Compare High vs. Low ITE Customers
summary_table <- df %>%
  group_by(ITE_group) %>%
  summarise(
    avg_age = mean(age, na.rm = TRUE),
    avg_balance = mean(balance, na.rm = TRUE),
    avg_duration = mean(duration, na.rm = TRUE),
    avg_campaign = mean(campaign, na.rm = TRUE),
    housing_loan_ratio = mean(housing == "yes", na.rm = TRUE),
    personal_loan_ratio = mean(loan == "yes", na.rm = TRUE)
  )

print(summary_table)

# Boxplot: How Features Differ Across ITE Groups
ggplot(df, aes(x = ITE_group, y = age, fill = ITE_group)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Age Distribution by ITE Group",
       x = "ITE Group",
       y = "Age")

ggplot(df, aes(x = ITE_group, y = balance, fill = ITE_group)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Balance Distribution by ITE Group",
       x = "ITE Group",
       y = "Balance")

# Regression Analysis: What Predicts ITE?
ite_model <- lm(ITE ~ age + balance + duration + campaign + housing + loan + job + marital + education + contact, data = df)

# Display regression results
summary(ite_model)

# Feature Importance: Which Variables Matter Most?
library(randomForest)

# Train a random forest model to predict ITE
rf_model <- randomForest(ITE ~ age + balance + duration + campaign + housing + loan + job + marital + education + contact, 
                         data = df, importance = TRUE, ntree = 500)

# Plot variable importance
varImpPlot(rf_model, main = "Feature Importance in Predicting ITE")

# Feature Analysis for Individual Treatment Effects (ITE) Reveals Key Patterns  

# Customers Who Benefit the Most from Previous Contact (High ITE)  
# - Higher balance customers tend to respond better to marketing efforts.  
# - Longer call duration is strongly linked to a higher probability of subscription.  
# - Married customers and those with higher education levels are more likely to subscribe after contact.  

# Customers Who Benefit the Least (Low ITE)  
# - Blue-collar, retired, and self-employed customers have significantly lower treatment effects.  
# - Customers contacted via telephone are less likely to respond positively compared to other contact methods.  
# - Shorter calls correlate with lower ITE, indicating lack of engagement leads to ineffective contact.  

# Key Takeaways for Targeting Strategy  
# - Prioritize high-balance, well-educated, and married customers with longer call durations.  
# - Rethink outreach to blue-collar, retired, and self-employed customers, as they show lower response rates.  
# - Improve call engagement strategies to increase conversion rates for lower ITE customers.  


############################
### Causal Random Forest ###
############################

# Load necessary libraries
library(dplyr)
library(grf)  # Generalized Random Forests for Causal Inference
library(ggplot2)

# Define treatment variable (previous contact)
df <- df %>%
  mutate(treated = ifelse(previous > 0, 1, 0))  # 1 = contacted before, 0 = never contacted

# Select features and outcome variable
features <- c("age", "balance", "duration", "campaign", "housing", "loan", "job", "marital", "education", "contact")
X <- df[, features]  # Covariates
W <- df$treated      # Treatment variable (previous contact)
Y <- df$y            # Outcome variable (subscription)

# Convert categorical variables to numeric for CRF
X <- model.matrix(~ . -1, data = X)  # One-hot encoding for categorical features

# **1️⃣ Train the Causal Random Forest Model**
crf_model <- causal_forest(X, Y, W)

# **2️⃣ Estimate Individual Treatment Effects (ITE)**
ITE_estimates <- predict(crf_model)$predictions
df$ITE_CRF <- ITE_estimates  # Store ITE estimates in dataset

# **3️⃣ Feature Importance Analysis**
feature_importance <- variable_importance(crf_model)

# Convert to data frame for visualization
feature_importance_df <- data.frame(
  Feature = colnames(X),
  Importance = feature_importance
) %>%
  arrange(desc(Importance))

# **4️⃣ Visualizing ITE Distribution**
ggplot(df, aes(x = ITE_CRF)) +
  geom_histogram(bins = 30, fill = "blue", alpha = 0.6) +
  theme_minimal() +
  labs(title = "Distribution of Estimated Individual Treatment Effects (ITE) - Causal RF",
       x = "Individual Treatment Effect (ITE)",
       y = "Count")

# **5️⃣ Visualizing Feature Importance**
ggplot(feature_importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Feature Importance in Causal Random Forest",
       x = "Feature",
       y = "Importance Score")

# Findings from Causal Random Forest Analysis

# Feature Importance in Treatment Effects  
# - Call duration is the most influential factor in determining subscription likelihood.  
# - Age, balance, and housing status also play significant roles in treatment effects.  
# - Contacting customers via telephone has a moderate impact, suggesting that contact method matters.  

# Distribution of Estimated Individual Treatment Effects (ITE)
# - Most customers have positive treatment effects, meaning previous contact generally increases subscription rates.  
# - A small portion of customers experience negative effects, indicating that outreach may backfire for certain groups.  
# - The distribution suggests variation in response, highlighting the importance of targeted marketing strategies.  

# Conclusion
# Our analysis confirms that previous contact increases subscription rates on average, but the effect varies by customer characteristics.  
# By leveraging causal machine learning, we identified key predictors of treatment effects, helping optimize future marketing efforts.  
```