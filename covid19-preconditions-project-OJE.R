######################################################################
#
# Project Instructions from the EdX ML Capstone course
#
######################################################################

# The Role of Pre-Existing Conditions in COVID-19 Outcomes

######################################################################
#
# The dataset
#
######################################################################

# COVID-19 Patient pre-conditions (44.52 MB).
# 
# from Kaggle (https://www.kaggle.com/ (https://www.kaggle.com/))
# The data was obtained from the "COVID-19 related cases dataset", released by the Mexican government. The original dataset is public,
# and can be downloaded from the page of the General Direction of Epidemiology, belonging to the Health Secretary of the Mexican
# government. https://www.gob.mx/salud/documentos/datos-abiertos-152127 (https://www.gob.mx/salud/documentos/datos-abiertos-152127)
 
# The data pertain to Mexican individuals.It contains a large number of anonymized patient information, including data about 12 medical conditions.
# The dataset has a total of 566602 observations(rows) and 23 features(columns) to describe each observation.


# Features description:
# 
# Id: patient id. Auto generated Random #. Age: Patient age in years
# The Sex definition was corrected:
# Sex: 1 - Male, 2 - Female, 99 - not specified (Incorrect definition, proven by data inconsistency detected)
# Sex: 1 - Female, 2 - Male, 99 - not specified (Correct definition. Details are explained in the Data Preparation section)
# Covid Res: Test Result. 1 - Positive SARS-CoV-2, 2 - Negative SARS-CoV-2, 3 - Pending Result
# Patient Type: 1 - ambulatory, 2 - hospitalized, 99 - not specified
# Date Died: Date, other values: 9999-99-99
# ICU: Patient required ICU admission. 1 - Yes, 2 - No, other values: 97, 98, 99
# Patient health Conditions:
#   List: pneumonia, pregnancy, diabetes, copd, asthma, immunosuppression, hypertension, cardiovascular, obesity, renal_chronic, tobacco,
# other_disease.
# Identifies if the patient was diagnosed with the Condition.
# 1 - Yes, 2 - No, other values: 97, 98, 99
# Other Values:
# Value  Meaning
# 97     NOT APPLICABLE
# 98     IGNORED
# 99     NOT SPECIFIED


######################################################################
#
# Key Steps from Data Cleaning to Model Evaluation
#
######################################################################

# Data Cleaning: Data loading, preparation, including handling missing data. Use of dplyr.
# Exploratory Data Analysis: Replicate summary statistics and data visualization (bar charts and pie charts) with ggplot2.
# Feature Engineering: Implement transformations and feature creation using mutate and logical conditions in dplyr.
# Model Building: Create ML models, training them on the training data, and creating predictions on the test data.
# Model Evaluation: Implement accuracy calculation and evaluate model performance using other relevant metrics from the caret package.

######################################################################
#
# Library loading 
#
######################################################################

if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if (!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if (!require(data.table)) install.packages("recipes", repos = "http://cran.us.r-project.org")
if (!require(data.table)) install.packages("themis", repos = "http://cran.us.r-project.org")


library(dplyr)
library(tidyverse)
library(data.table)
library(ggplot2)
library(caret)
library(rpart)
library(randomForest)


######################################################################
#
# Data Loading
#
######################################################################


# TODO: First, the covid19 dataset files needs to be downloaded from the source below.
# https://www.kaggle.com/api/v1/datasets/download/tanmoyx/covid19-patient-precondition-dataset

# NOTE: For convenience, the dataset files are provided along with the project files. 

# Assumption: At this point, it is assumed that the dataset was downloaded and its content extracted to the directory "covid19-patients-preconditions-dataset", 
# located in the same directory as this R script file.

# Set the working directory to the current directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) 

# Get current working directory to make sure the dataset can be loaded
getwd()

# List files in the dataset files directory
list.files("covid19-patients-preconditions-dataset")

# The data for this dataset in a single file, which will be used to answer the research questions for this project.
# covid.csv
# In the folder, there is also a file with the data dictionary and another file with the catalogs that describe the data in the dataset.
# Description.xlsx
# Catalogs.xlsx

# Load the data into a data frame
# Load the file 'covid.csv' that contains the patients data to be utilized in this project
covid19_preconditions <- fread("covid19-patients-preconditions-dataset/covid.csv")

# Read the CSV file into a data frame
# read_csv() from the readr package (part of tidyverse) is used to accomplish this task
# covid19_preconditions <- read_csv("covid19-patients-preconditions-dataset/covid.csv", show_col_types = FALSE)

class(covid19_preconditions) 

######################################################################
#
# Data Exploration and Preparation
#
######################################################################

############################
# Initial data exploration

# Data Summary and Structure
# To get a summary and check the structure, similar to data frame.describe() and .head() in Python:

# Dataset Overview
str(covid19_preconditions) # Data Shape compactly displayed
summary(covid19_preconditions)  # Summary statistics
head(covid19_preconditions) # first few rows of the data frame

# This dataset has a total of 566602 observations(rows) and 23 features(columns) to describe each observation.

# Missing Values
# Per the summary and sample data presented above, it looks like the dataset doesn't have any explicit missing value. 
# However, by looking into the data dictionary and catalogs files, it can be determined that the missing data has been encoded with these specific values:
# 97 - NOT APPLICABLE , 98 - IGNORED, 99 - NOT SPECIFIED
# This means other operations will be needed in order to better identify unwanted missing data, and remove it as needed.


# Detect data inconsistencies

# The dplyr package was utilized to filter the data based on specific conditions.

total_men <- nrow(covid19_preconditions |> filter(sex == 1))
total_women <- nrow(covid19_preconditions |> filter(sex == 2))

cat("Total men:", total_men)
cat("Total women:", total_women)

# Detect observations of women with pregnancy = 2(No)
non_pregnant_women <- covid19_preconditions |> filter(sex == 2, pregnancy == 2)
if (nrow(non_pregnant_women) > 0) {
  print("There are " , nrow(non_pregnant_women), "observations of women with pregnancy = 2 (No)")
}

# Detect observations of women with pregnancy = 1(Yes)
pregnant_women <- covid19_preconditions |> filter(sex == 2, pregnancy == 1)
if (nrow(pregnant_women) > 0) {
  print("There are " , nrow(pregnant_women), "observations of women with pregnancy = 1 (Yes)")
}

# Detect observations of men with pregnancy = 1(Yes)
pregnant_men <- covid19_preconditions |> filter(sex == 1, pregnancy == 1)
if (nrow(pregnant_men) > 0) {
  cat("There are" , nrow(pregnant_men), "observations of men with pregnancy = 1 (Yes)")
}

# Analyze pregnancy values in women
# The data is filtered to keep only rows with sex equal to 2(female).
# Count the number of women with unknown pregnancy values (i.e., pregnancy is NA).

women_data <- covid19_preconditions |> filter(sex == 2)
head(women_data)
unknown_pregnancy_women <- women_data |> filter(pregnancy %in% c(97, 98, 99) | is.na(pregnancy))
na_pregnancy_women <- women_data |> filter(pregnancy == 97) # the code 97 corresponds to NOT APPLICABLE

cat("Total women with unknown pregnancy:", nrow(unknown_pregnancy_women))
cat("Total women with pregnancy = NOT APPLICABLE:", nrow(na_pregnancy_women))
cat("All the records from women in the dataset", nrow(women_data) ,"have pregnancy = NOT APPLICABLE")

# Conclusion: 
# All pregnancy values for women in this dataset are unknown (represent missing data), while the observations corresponding to men have valid pregnancy values.
# This indicates there is an error in the data. It looks like either the "pregnancy" or the "sex" feature was flipped, causing these inconsistencies.  
# All these facts indicate that there is a mistake in the definition of the sex column.

# After downloading the original descriptors for the dataset from the Mexican government web page, 
# it was confirmed that "sex" should be switched: sex = 1 is female, and sex = 2 is male.


# Detect observations with patients not hospitalized (ambulatory) and admitted to ICU
ambulatory_icu <- covid19_preconditions |> filter(patient_type == 1, icu == 1)

if (nrow(ambulatory_icu) == 0) {
  cat("- There are no observations of ambulatory patients admitted to an ICU.", "\n")
}
# Conclusion: There are no inconsistencies in the data related to patient type and hospitalizations. 

# Detecting observations with patients that tested Negative for COVID-19 and were hospitalized

# filter the data by patient type and covid test result
no_covid_inpatient <- covid19_preconditions |> filter(covid_res == 2, patient_type == 2)

if (nrow(no_covid_inpatient) > 0) {
  cat("There are observations corresponding to non-COVID-19 patients that were hospitalized", "\n")
}

# The dataset contains data from patients that were hospitalized for reasons other than a COVID-19 diagnosis.
# This is not a data inconsistency, but it is important to take it into account in the data analysis, 
# in case these observations are not relevant at some point and need to be excluded.


############################
# Data Preparation

# Handling Missing Values
# The dataset used values 97, 98, and 99 to represent missing data. 

# Replace missing values encoded as 97, 98, and 99 with NA

# Check for missing values
# This calculates the number of missing values in each column of the data frame.
colSums(is.na(covid19_preconditions))

# Utilized this approach which is less costly in terms of machine power:
covid19_preconditions[covid19_preconditions == 97] <- NA
covid19_preconditions[covid19_preconditions == 98] <- NA
covid19_preconditions[covid19_preconditions == 99] <- NA

# Remove duplicates
covid19_preconditions <- covid19_preconditions |> distinct()
dim(covid19_preconditions)

# Filtering Confirmed COVID-19 Cases
# Filter out the observations where covid_res is equal to 1 (confirmed cases) and then remove the covid_res column.
covid19_confirmed_patients <- covid19_preconditions |>
  filter(covid_res == 1) |>
  select(-covid_res)  # Remove covid_res column  for "Covid Result" as it's no longer needed

dim(covid19_confirmed_patients)

# Feature Engineering

# Apply transformations and create new features using mutate and logical conditions with dplyr.

# Select Relevant Columns and Discarding Irrelevant Data
# Exclude columns that aren’t directly related to the analysis (irrelevant_features), keeping only those in relevant_features.

irrelevant_features <- c("id", "entry_date", "date_symptoms", "contact_other_covid", "intubed")
relevant_features <- c("age", "sex", "patient_type", "icu", "date_died", 
                       "pneumonia", "pregnancy", "diabetes", "copd", "asthma", "inmsupr", 
                       "hypertension", "cardiovascular", "obesity", "renal_chronic", 
                       "tobacco", "other_disease")

# Select only relevant columns
covid19_confirmed_patients <- covid19_confirmed_patients |>
  select(all_of(relevant_features))

dim(covid19_confirmed_patients)


# Rename Columns

# Rename the patient_type column to inpatient and the icu column to required_icu
covid19_confirmed_patients <- covid19_confirmed_patients |>
  rename(inpatient = patient_type, required_icu = icu)

# Decode and Transform Columns to Boolean
# convert the sex, inpatient, and required_icu columns to make them more interpretable, and transform the conditions_features to logical (boolean) values for analysis:

# Decoding the Sex feature for easier recognition. `1` = "F" and `2` = "M"
covid19_confirmed_patients <- covid19_confirmed_patients |> mutate(sex = factor(sex, labels = c("F", "M")))

# verify the data now makes sense
women_after_transform_data <- covid19_confirmed_patients |> filter(sex == "F")
print(count(women_after_transform_data))
unknown_pregnancy_women <- women_after_transform_data |> filter(pregnancy %in% c(97, 98, 99) | is.na(pregnancy))
na_pregnancy_women <- women_after_transform_data |> filter(pregnancy == "TRUE") # the code 97 corresponds to NOT APPLICABLE

cat("Total women with unknown pregnancy:", nrow(unknown_pregnancy_women))
cat("Total women with pregnancy = NOT APPLICABLE:", nrow(na_pregnancy_women))


#Transform the data to Boolean for easier manipulation

# Transform all the features related to patient health conditions, to boolean values (TRUE for 1, FALSE for 2)
conditions_features <- c("pneumonia", "pregnancy", "diabetes", "copd", "asthma", "inmsupr", "hypertension", "cardiovascular", "obesity", "renal_chronic", "tobacco", "other_disease")

# Convert columns to logical
# The properties inpatient and required_icu the values are directly compared to 2 and 1, respectively, and assign TRUE or FALSE accordingly.
# For the conditions_features, use across() to apply the same transformation to all specified columns. 
# The lambda function ~ .x == 1 checks if each value is equal to 1 and assigns TRUE or FALSE.

covid19_confirmed_patients <- covid19_confirmed_patients |>
  mutate(
    inpatient = inpatient == 2,
    required_icu = required_icu == 1,
    across(all_of(conditions_features), ~ .x == 1)
  )

# Display the first few rows
head(covid19_confirmed_patients)


# Add new columns to the data frame for further analysis:
# has_preconditions(boolean): if the patient has pre-existing conditions. This is a representation of whether or not the patient has risk factors. 
# confirmed_deceased(boolean): if the patient is deceased  
# confirmed_icu(boolean): if the patient was admitted to an ICU

covid19_confirmed_patients <- covid19_confirmed_patients |>
  mutate(
    has_preconditions = (pneumonia == TRUE & !is.na(pneumonia)) |
      (diabetes == TRUE & !is.na(diabetes)) |
      (pregnancy == TRUE & !is.na(pregnancy)) |
      (copd == TRUE & !is.na(copd)) |
      (asthma == TRUE & !is.na(asthma)) |
      (inmsupr == TRUE & !is.na(inmsupr)) |
      (hypertension == TRUE & !is.na(hypertension)) |
      (cardiovascular == TRUE & !is.na(cardiovascular)) |
      (obesity == TRUE & !is.na(obesity)) |
      (renal_chronic == TRUE & !is.na(renal_chronic)) |
      (tobacco == TRUE & !is.na(tobacco)) |
      (other_disease == TRUE & !is.na(other_disease)),
    confirmed_deceased = if_else(date_died == "9999-99-99", FALSE, TRUE),
    confirmed_icu = if_else(required_icu == TRUE, TRUE, FALSE)
  )

# Check for NA values in all features
unique(covid19_confirmed_patients$has_preconditions)

# Display the first few rows
head(covid19_confirmed_patients)

# The next step is exploring and visualizing the most relevant aspects of this dataset (e.g., health conditions and distributions). 

# Total observations in the dataset
total_observations <- nrow(covid19_preconditions)
cat("Total observations in the dataset:", total_observations, "\n")

# Confirmed cases of Covid-19
cat("Total observations of Confirmed cases of Covid-19 in the dataset:", nrow(covid19_confirmed_patients), "\n")

# Sex distribution
cat("Total women confirmed cases of Covid-19:", nrow(covid19_confirmed_patients |> filter(sex == 'F')))
cat("Total men confirmed cases of Covid-19:", nrow(covid19_confirmed_patients |> filter(sex == 'M')))


######################################################################
#
# Data Analysis and Visualization
#
######################################################################

# Replicate summary statistics and data visualization (bar charts and pie charts) with ggplot2.

# In this step some calculations will be performed on the dataset in order to quantify the variables needed for the analysis.
# This will be complemented with data visualization techniques to aid gaining insights on the data.

# These techniques will address the first three research questions by comparing death rates, hospitalization rates,
# and the prevalence of underlying conditions.

# To help answer the first three Research Questions formulated for this project, related to deaths and hospitalizations rates:
#   - Are individuals with underlying health conditions more likely to die from COVID-19 compared to healthy people?
#   - Are COVID-19 patients with underlying health conditions more vulnerable to becoming severely ill with the virus, requiring hospitalization and intensive care?
#   - What are the most common underlying conditions in COVID-19 patients?

# Data visualizations:

# Pie Charts for overall case distribution and health condition distribution.
# Bar Charts for hospitalization and death rates, categorized by underlying conditions.
# Bar Chart showing the most common pre-existing health conditions.
# Bar Chart of Common Underlying Conditions


# Analyzing COVID-19 patients case distribution

# Calculate COVID-19 case distribution
total_confirmed_cases <- nrow(covid19_confirmed_patients)
negative_pending_cases <- total_observations - total_confirmed_cases
cat("Total confirmed COVID-19 cases in the dataset:", total_confirmed_cases, "\n")

# Calculate percentage of confirmed COVID-19 cases
percentage_confirmed_cases <- (total_confirmed_cases / total_observations) * 100
cat("Total observations of confirmed COVID-19 cases in the dataset:", total_confirmed_cases, "(", round(percentage_confirmed_cases, 2), "%)\n")

# Data for plotting
case_distribution <- data.frame(
  Status = c("Confirmed Positive", "Negative or Pending Results"),
  Count = c(total_confirmed_cases, negative_pending_cases)
)

#  COVID-19 Case Distribution Pie Chart
# This chart shows the proportion of confirmed cases versus negative/pending cases.
ggplot(case_distribution, aes(x = "", y = Count, fill = Status)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y") +
  theme_void() +
  labs(title = "COVID-19 Cases Distribution") +
  geom_text(aes(label = scales::percent(Count / sum(Count), accuracy = 0.01)), position = position_stack(vjust = 0.5))



# Analyzing the patients with and without Health Conditions Pie Chart

# Calculate counts for patients with and without underlying conditions
with_conditions <- covid19_confirmed_patients |> filter(has_preconditions == TRUE)
total_with_conditions <- nrow(covid19_confirmed_patients |> filter(has_preconditions == TRUE)) 
total_without_conditions <- nrow(covid19_confirmed_patients |> filter(has_preconditions == FALSE))

# Calculate percentage of confirmed COVID-19 cases with health conditions
percentage_with_conditions <- (total_with_conditions / total_confirmed_cases) * 100
cat("Total observations of confirmed COVID-19 cases with health conditions:", total_with_conditions, "(", round(percentage_with_conditions, 2), "%)\n")

# Calculate percentage of confirmed COVID-19 cases without health conditions
percentage_without_conditions <- (total_without_conditions / total_confirmed_cases) * 100
cat("Total observations of confirmed COVID-19 cases without health conditions:", total_without_conditions, "(", round(percentage_without_conditions, 2), "%)\n")


# Data for plotting
precondition_distribution <- data.frame(
  Status = c(" COVID-19 Patients with Underlying Conditions", " COVID-19 Patients Without Underlying Conditions"),
  Count = c(total_with_conditions, total_without_conditions)
)

# Pie chart
# This pie chart shows the proportion of COVID-19 patients with at least one underlying health condition versus those without.
ggplot(precondition_distribution, aes(x = "", y = Count, fill = Status)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y") +
  theme_void() +
  labs(title = "Distribution of Underlying Health Conditions in COVID-19 Patients") +
  geom_text(aes(label = scales::percent(Count / sum(Count), accuracy = 0.01)), position = position_stack(vjust = 0.5))



# Analyzing patients hospitalization and death rates

# To answer the questions about death and hospitalization rates among those with and without underlying conditions,
# bar charts were created to compare these rates.

# A: Hospitalizations and Deaths for Patients with Underlying Conditions

# Calculate hospitalization and death rates for patients with underlying conditions
hospitalized_with_conditions <- sum(covid19_confirmed_patients$inpatient & covid19_confirmed_patients$has_preconditions)
deaths_with_conditions <- sum(covid19_confirmed_patients$confirmed_deceased & covid19_confirmed_patients$has_preconditions)

# Calculate percentage of confirmed COVID-19 cases with health conditions that required hospitalization
percentage_hospitalized_with_conditions <- (hospitalized_with_conditions * 100 / total_with_conditions)
cat("Total observations of confirmed COVID-19 cases with health conditions that required hospitalization (Including ICU):", hospitalized_with_conditions, "(", round(percentage_hospitalized_with_conditions, 2), "%)\n")

# Calculate percentage of confirmed COVID-19 cases with health conditions that died
percentage_deaths_with_conditions <- (deaths_with_conditions * 100 / total_with_conditions) 
cat("Total observations of confirmed COVID-19 cases with health conditions that died:", deaths_with_conditions, "(", round(percentage_deaths_with_conditions, 2), "%)\n")

# Data for plotting
hospitalization_death_with_conditions <- data.frame(
  Status = c("Required Hospitalization\n(Including ICU)", "Confirmed Deaths"),
  Count = c(hospitalized_with_conditions, deaths_with_conditions)
)

# Horizontal bar chart
ggplot(hospitalization_death_with_conditions, aes(x = Count, y = Status)) +
  geom_bar(stat = "identity", width = 0.8, fill = c("#6259D8", "#E53F08")) +  # Set bar width and colors
  labs(title = "Hospitalizations and Deaths among COVID-19 Patients with underlying health conditions", 
       x = "Number of patients",
       y = "") +  # Remove label for y-axis
  scale_x_continuous(breaks = seq(0, total_with_conditions, by = 10000)) + # Set breaks every 10000 units
  geom_text(aes(label = paste0(Count, " (", scales::percent(Count / total_with_conditions, accuracy = 0.01), ")")), position = position_stack(vjust = 0.5), color = "white") +
  theme_minimal() +  # Use minimal theme aesthetics
  theme(axis.text.y = element_text(face = "bold"),
        legend.position = "none")  # Remove legend


# B: Hospitalizations and Deaths for Patients without Underlying Conditions

# Calculate hospitalization and death rates for patients without underlying conditions
hospitalized_without_conditions <- sum(covid19_confirmed_patients$inpatient & !covid19_confirmed_patients$has_preconditions)
deaths_without_conditions <- sum(covid19_confirmed_patients$confirmed_deceased & !covid19_confirmed_patients$has_preconditions)

# Calculate percentage of confirmed COVID-19 cases with health conditions that required hospitalization
percentage_hospitalized_without_conditions <- (hospitalized_without_conditions * 100 / total_without_conditions)
cat("Total observations of confirmed COVID-19 cases with health conditions that required hospitalization (Including ICU):", hospitalized_without_conditions, "(", round(percentage_hospitalized_without_conditions, 2), "%)\n")

# Calculate percentage of confirmed COVID-19 cases with health conditions that died
percentage_deaths_without_conditions <- (deaths_without_conditions * 100 / total_without_conditions) 
cat("Total observations of confirmed COVID-19 cases with health conditions that died:", deaths_without_conditions, "(", round(percentage_deaths_without_conditions, 2), "%)\n")

# Data for plotting
hospitalization_death_without_conditions <- data.frame(
  Status = c("Required Hospitalization\n(Including ICU)", "Confirmed Deaths"),
  Count = c(hospitalized_without_conditions, deaths_without_conditions)
)

# Horizontal bar chart
ggplot(hospitalization_death_without_conditions, aes(x = Count, y = Status)) +
  geom_bar(stat = "identity", width = 0.8, fill = c("#6259D8", "#E53F08")) +  # Set bar width and colors
  labs(title = "Hospitalizations and Deaths among COVID-19 Patients without underlying health conditions",
       x = "Number of patients",
       y = "") +  # Remove label for y-axis
  scale_x_continuous(breaks = seq(0, total_without_conditions, by = 2000)) + # Set breaks every 2000 units
  geom_text(aes(label = paste0(Count, " (", scales::percent(Count / total_without_conditions, accuracy = 0.01), ")")), position = position_stack(vjust = 0.5), color = "white") +
  theme_minimal() +  # Use minimal theme aesthetics
  theme(axis.text.y = element_text(face = "bold"),
        legend.position = "none")  # Remove legend



# Discovery of the most common underlying conditions

# This final chart identifies the most common health conditions among confirmed COVID-19 patients, answering the third question.

# Count occurrences of each condition
condition_counts <- covid19_confirmed_patients |>
  select(all_of(conditions_features)) |>
  summarize(across(everything(), ~ sum(.x, na.rm = TRUE)))

# Convert to data frame with condition names and counts for plotting
condition_counts_df <- data.frame(
  Condition = names(condition_counts),
  Count = as.integer(condition_counts)
)

print(condition_counts_df)

# Sort conditions by frequency
condition_counts_df <- condition_counts_df |> arrange(desc(Count))


# visualize the distribution of medical conditions in COVID-19 patients.
# This chart shows each medical condition on the x-axis and the number of patients who reported the condition on the y-axis.

#retrieve the names of the predefined palettes
palette.pals()

# Bar chart
ggplot(condition_counts_df, aes(x = reorder(Condition, -Count), y = Count, fill = Condition)) +
  geom_bar(stat = "identity") +
  labs(title = "Pre-existing Conditions Distribution in COVID-19 Patients", x = "Condition", y = "Number of Patients") +
  scale_fill_brewer(palette = "Set3") +  # Use a color palette for random colors
  scale_y_continuous(breaks = seq(0, total_confirmed_cases, by = 10000)) + # Set breaks every 10000 units
  geom_text(aes(label = format(Count, big.mark = ",")), vjust = -0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"), 
        axis.text.y = element_text(face = "bold"),
        legend.position = "none")  # Remove legend)



# Insights gained:
# 
# Per the graphic above the most common underlying conditions, reported by more than 30,000 COVID-19 patients in Mexico, were
# pneumonia, hypertension, obesity and diabetes. The next most common is tobacco, reported by 17,109 patients. Less than 10,000
# patients reported other pre-existing conditions.
# The U.S. Centers for Disease Control and Prevention (CDC) June 2020 COVID-19 report, states that the most common underlying
# conditions reported in COVID-19 patients in the United States were heart disease, diabetes and chronic lung disease.
# Although it is logical that the most common conditions vary by country, it can be seen that there are some such as diabetes and heart
# disease, which are a common denominator between countries.


######################################################################
#
# Data Modeling
#
######################################################################

# In order to check the factibility of predicting if a COVID-19 patient is going to require admission to an intensive care unit(ICU) based on his
# underlying medical conditions, the “classification” machine learning technique will be applied to the dataset.

# -------------------------
# Steps in the modeling phase:
#
# 1  Data Cleaning and Preparation:
# Select relevant classification features.
# Remove rows with missing values in the target feature "confirmed_icu"

# 2 Train-Test Split
# The data will be divided into 70% training and 30% testing subsets.

# 3 Model building
# Build a model for binary classification on ICU admission.
# Evaluate model performance on a test set.
# - Decision Tree Model
# - Random Forests Model
# - Gradient Boosting Machines (GBM) Model

# 4 Model Evaluation: Implement accuracy calculation and evaluate model performance.
# The accuracy of the models was calculated using the "accuracy" metric, defined as the proportion of correctly classified observations over the total number of observations.
# This measurement was computed as:

# Accuracy = "Number of Correct Predictions" / "Total Number of Observations"

# This metric is straightforward to calculate and understand, and it provides an overall measure of how well the model is performing. Here's why accuracy was chosen:

# Why Accuracy?
#   Binary Classification Task:

#   This is a binary classification problem where the target variable is confirmed_icu, has two possible outcomes (TRUE for requiring ICU and FALSE for not requiring ICU).
# For such problems, accuracy is a commonly used metric to evaluate performance.

# Balanced Dataset Assumption:

#   If the dataset has a relatively balanced number of cases for both classes (e.g., patients requiring ICU vs. not requiring ICU), accuracy is a good starting point to evaluate the model.

# Baseline Understanding:

#   Accuracy provides a quick, high-level understanding of the model's performance, making it useful for initial comparisons between models (e.g., decision tree vs. random forest).

# -------------------------


# Step 1: Data Cleaning and Preparation
# Select the relevant features for this classification problem 
# Remove rows with missing values in the confirmed_icu feature

# Load necessary libraries
# library(dplyr)

nrow(covid19_confirmed_patients)
head(covid19_confirmed_patients)

# Define classification features and filter relevant data
classification_features <- c("confirmed_icu", "has_preconditions", "pneumonia", "pregnancy", 
                             "diabetes", "copd", "asthma", "inmsupr", "hypertension", 
                             "cardiovascular", "obesity", "renal_chronic", "tobacco", "other_disease")

print(paste("Total Observations before cleaning:", nrow(covid19_confirmed_patients)))

print(nrow(covid19_confirmed_patients |> filter(confirmed_icu == TRUE))) # 5822

# Select classification data per the selected features
classification_data <- covid19_confirmed_patients |>
  select(all_of(classification_features))

class(classification_data)
head(classification_data)

# Check the features that have missing values
unique(classification_data) 
any_na <- any(is.na(classification_data))
print(paste("Are there any missing values remaining in the data frame?", any_na))

# Remove Missing Values from the data set
# The goal here is to drop rows with any missing values in the classification_data dataset and check the before-and-after row counts to ensure proper cleaning.

# Drop rows with missing values in any column
# This includes removing the rows with NA in the confirmed_icu column which is the target variable
classification_clean_data <- classification_data |> drop_na()

#cat(paste("Total Observations after dropping observations with missing values in the target feature confirmed_icu:", nrow(classification_data))) # 68210
cat("Total of observations from patients that were admitted to an ICU:", nrow(classification_data |> filter(confirmed_icu == TRUE)), "\n") # 5822

# Check the number of rows after removing NA values
#after_rows <- nrow(classification_clean_data)
#print(paste("Rows after cleaning:", after_rows)) # 25087

#print(nrow(classification_data |> filter(confirmed_icu == TRUE)))
#unique(classification_data$confirmed_icu) # FALSE TRUE

# Confirm there are no remaining NA values in the dataset
any_na <- any(is.na(classification_clean_data))
print(paste("Are there any missing values remaining in the data frame?", any_na))

# Re confirm no missing values are present in the target feature/variable
unique(classification_clean_data$confirmed_icu) # FALSE TRUE

# Print Row Counts:
#   Review the number of observations before and after cleaning to understand the impact of removing missing values.

# Print the number of rows before and after cleaning
cat("The classification data frame has", ncol(classification_data), "features", "\n")
cat("Total observations before cleaning:", nrow(classification_data), "\n")
cat("Total observations after cleaning:", nrow(classification_clean_data), "\n") # 25087 , and 67300 without pregnancy included

cat("After cleaning, there are a total of", nrow(classification_clean_data |> filter(confirmed_icu == TRUE)), "observations from patients that were admitted to an ICU", "\n")

# Set cleaned data back into classification_data
classification_data <- classification_clean_data


# Inspecting the quality of the data, since this is a will affect/determine the model's selection
# Common data issues are that the features(independent variables) might have constant or irrelevant values, or the target variable is poorly balanced.

# Check Data Balance
# Check the distribution of the target variable (confirmed_icu):

table(classification_data$confirmed_icu)
# FALSE  TRUE 
# 23223  1864 

# Insights gained: The target variable confirmed_icu is heavily imbalanced, with the majority of cases being FALSE.
# Consider balancing the dataset using techniques like oversampling or undersampling.

# Inspect Features
# Check for low variance or irrelevant features in the data:

summary(classification_data)

# Feature Engineering

# Perform further feature selection based on the quality of the data, and the correlation with the target variable confirmed_icu.
# Consider to Remove features that have little or no variation (e.g., all values are the same).

# Evaluation of the features and recommendations for discarding irrelevant or low-variance features.

# Target Variable (confirmed_icu): This variable is imbalanced, with FALSE: 23223 and TRUE: 1864. This class imbalance may hinder model performance.


# The correlation values indicate the strength of the linear relationship between each feature and the target variable (confirmed_icu). 
# To verify feature importance using correlation with the target variable confirmed_icu,
# calculate the point-biserial correlation between the target (binary) and each feature (binary or logical). 

# Convert logical variables to numeric
classification_data_numeric <- classification_data |>
  mutate(across(everything(), as.numeric))

# Calculate correlations with the target variable
correlations <- sapply(classification_data_numeric[, -1], function(x) cor(x, classification_data_numeric$confirmed_icu, use = "complete.obs"))

# Sort correlations by absolute value
sorted_correlations <- sort(correlations, decreasing = TRUE)

# Print sorted correlations
print(sorted_correlations)

# The correlation values indicate the strength of the linear relationship between each feature and confirmed_icu:
# Values close to 1 or -1: Strong correlation (high importance).
# Values close to 0: Weak correlation (low importance, potentially irrelevant).


# Feature Ranking

# Highly Relevant Features:
# pneumonia (0.125): Strongest correlation. Retain.
# has_preconditions (0.065): Moderately correlated. Retain.

# Moderately Relevant Features:
# obesity (0.035): Weak correlation, but retain as it might interact with other conditions.
# diabetes (0.018): Weak but potentially important in a medical context. Retain.

# Weakly Relevant Features (Consider Discarding):
# inmsupr (0.012): Minimal correlation; discard unless domain knowledge suggests relevance.
# cardiovascular (0.011): Similar to inmsupr, likely discard.
# hypertension (0.010): Weak, but retain due to its prevalence in ICU patients.

# Least Relevant Features (Discard):
# pregnancy (0.007): Negligible correlation, likely discard.
# other_disease (0.001): Very weak, discard.
# copd (0.0007): Very weak, discard.
# renal_chronic (-0.00002): No meaningful relationship, discard.
# asthma (-0.0001): No meaningful relationship, discard.
# tobacco (-0.015): Weak negative correlation, discard.


# Based on the results above, there are the features that will be retained to be used in the modeling phase:

# More Relevant and informative features:
# pneumonia
# has_preconditions
# obesity
# diabetes
# hypertension


# Update the dataset to remove the less relevant features.
classification_data <- classification_data |> 
  select(confirmed_icu, pneumonia, has_preconditions, obesity, diabetes, hypertension)


# Conclusion: 
# It was confirmed that the data is imbalanced. In this scenario, the decision tree model won't work because it will struggle splitting the data to improve the classification.
# Therefore, it was decided to use a different model that handles imbalanced data. The selected model was Random Forest, because it automatically balances decision trees across multiple subsets.

# Now, with clean and relevant data in the data frame, the next step is to build the classification model.



# Step 2: Split the data 
# Split the data into training and testing sets. The data will be divided into 70% training and 30% testing subsets.

# Split data into training and test sets (70% train, 30% test)
set.seed(324) # To ensure reproducibility
train_index <- createDataPartition(classification_data$confirmed_icu, p = 0.7, list = FALSE)
train_data <- classification_data[train_index, ]
test_data <- classification_data[-train_index, ]


# Check the class distribution before oversampling
table(train_data$confirmed_icu)



# Balance the training data

# It was confirmed before, the target Variable (confirmed_icu) is highly imbalanced, with FALSE: 23223(majority class) and TRUE: 1864(minority class).
# To avoid poor model performance, and improve its ability to learn from the minority class the training data will be balanced using oversampling.
# The ROSE package will be used for this task.


# TRUE and FALSE are logical values in R, and logical values cannot be used directly as class labels in many machine learning models.
# Renaming them to "Yes" and "No" ensures that they are treated as valid factor levels that can be used for classification.
# Convert the target variable to a factor with valid levels
train_data$confirmed_icu <- factor(train_data$confirmed_icu, 
                                   levels = c("FALSE", "TRUE"), 
                                   labels = c("No", "Yes"))

# Convert the target variable in test_data to match the levels in train_data
test_data$confirmed_icu <- factor(test_data$confirmed_icu, 
                                  levels = c("FALSE", "TRUE"), 
                                  labels = c("No", "Yes"))

library(ROSE)

# Oversample the minority class in training data
balanced_train_data <- ovun.sample(confirmed_icu ~ ., data = train_data, method = "over", N = 2 * max(table(train_data$confirmed_icu)))$data

# Check the class distribution after oversampling
table(balanced_train_data$confirmed_icu)


# Check for missing values in the training data
sum(is.na(balanced_train_data))
summary(balanced_train_data)



# Step 3: Model Building

# Data Classification using Decision Trees.


# Model #1: Decision Tree Model

# Use the rpart package for decision tree creation and testing.

# Build the decision tree model
icu_tree <- rpart(confirmed_icu ~ ., data = balanced_train_data, method = "class",
                   control = rpart.control(minsplit = 2, cp = 0.001, maxdepth = 30))
# Plot the tree
# plot(icu_tree)
# text(icu_tree, use.n = TRUE)

# Check if the tree splits
# print(icu_tree)


# Model Evaluation
# Use the test set to make predictions and then evaluate the accuracy.

# Predict ICU admission on test data
icu_predictions <- predict(icu_tree, test_data, type = "class")

# Calculate accuracy
accuracy <- sum(icu_predictions == test_data$confirmed_icu) / nrow(test_data)
print(paste("Decision Tree Accuracy:", round(accuracy * 100, 2), "%"))




# Model #2: Random Forests Model
# The Random Forest model often performs well on classification tasks by combining multiple decision trees. 

# Model Overview: Random Forest is an ensemble learning method that combines multiple decision trees to improve predictive accuracy and reduce overfitting. 
# It works by creating multiple decision trees on different subsets of the training data and then averaging the predictions of all the trees


# Load necessary library
library(randomForest) # Provides functions for creating random forest models.

# Ensure the target variable is a factor so the model knows this is a classification task.
# If the response vector is a factor, classification is assumed, otherwise regression is assumed. If omitted, randomForest will run in unsupervised mode.
# class(balanced_train_data$confirmed_icu)
# class(test_data$confirmed_icu)

# Train the Random Forest model on the balanced data
# randomForest() implements Breiman's random forest algorithm (based on Breiman and Cutler's original Fortran code) for classification and regression.

# Penalize misclassification of the minority class by setting class weights:
# Assign class weights inversely proportional to their frequency
class_weights <- c(
  "No" = 1 / table(balanced_train_data$confirmed_icu)["No"],
  "Yes"  = 1 / table(balanced_train_data$confirmed_icu)["Yes"]
)

# The names in the class_weights should match the levels in the target variable confirmed_icu
names(class_weights) <- levels(balanced_train_data$confirmed_icu)
levels(balanced_train_data$confirmed_icu)
print(class_weights)

# Train Random Forest model with class weights
set.seed(324)
icu_rf <- randomForest(confirmed_icu ~ ., data = balanced_train_data, ntree = 100, mtry = 3, classwt = class_weights)

# Predict on test data
rf_predictions <- predict(icu_rf, test_data)

# Calculate accuracy
rf_accuracy <- sum(rf_predictions == test_data$confirmed_icu) / nrow(test_data)
print(rf_accuracy)
print(paste("Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%"))


######################################################################
#
# Preliminary Results Summary
#
######################################################################

# The decision tree provides an interpretable model for understanding key predictors of ICU admission.
# The random forest offers improved predictive performance by leveraging ensemble learning.


# Potential Limitations of the Accuracy metric.

# Accuracy alone might not be the best metric if:

# Class Imbalance: If one class (e.g., FALSE) significantly outweighs the other (e.g., TRUE), 
# accuracy might be misleading because the model could predict the majority class most of the time and still appear to perform well.

# Importance of False Positives/Negatives: If the cost of false positives (predicting ICU when not needed) or false negatives (missing ICU patients) is high, 
# additional metrics like precision, recall, or F1-score would be more informative.



######################################################################
#
# Additional Metrics
#
######################################################################

# Accuracy is a good starting point to evaluate the models, if the dataset has a relatively balanced number of cases for both classes (e.g., patients requiring ICU vs. not requiring ICU),
# Since in this case the dataset is unbalanced, other metrics are needed to analyze the model's performance more thoroughly.

# Compute the confusion matrix, precision, recall, and F1-score for both the Decision Tree and Random Forest models.
# These metrics provide deeper insight into the model's handling of each class, especially if the dataset is imbalanced and the cost of errors is critical. 
# This is the case of medical applications where accurate prediction of critical outcomes is essential.

# - Confusion Matrix: Analyze the confusion matrix to see the distribution of true positives, false positives, true negatives, and false negatives.

# - Additional Metrics: Calculate and report precision, recall, and F1-score to provide a more nuanced evaluation of the models.



# The caret::confusionMatrix function is used to calculate confusion matrices for the Decision Tree and Random Forest models.

# Confusion Matrix for Decision Tree
 dt_conf_matrix <- confusionMatrix(data = as.factor(icu_predictions), 
                                  reference = as.factor(test_data$confirmed_icu))

# Confusion Matrix for Random Forest
rf_conf_matrix <- confusionMatrix(data = as.factor(rf_predictions), 
                                  reference = as.factor(test_data$confirmed_icu))

# Print confusion matrices
cat("Decision Tree Confusion Matrix:\n")
print(dt_conf_matrix)

cat("\nRandom Forest Confusion Matrix:\n")
print(rf_conf_matrix)


# Other Metrics (Precision, Recall, F1-Score)

# Calculations:
#   Precision: Proportion of predicted positives that are actual positives. Precision = TP / (TP + FP)
#   Recall (Sensitivity): Proportion of actual positives that are correctly identified. Recall = TP / (TP + FN)
#   F1-Score: Harmonic mean of precision and recall. F1-Score = 2 x ((Precision × Recall) / (Precision + Recall))


# Calculate Precision, Recall, and F1-Score using the confusion matrix output.
# These metrics are extracted from the confusion matrix for each model.

# Extract metrics from Decision Tree Confusion Matrix
dt_precision <- dt_conf_matrix$byClass["Pos Pred Value"]  # Precision
dt_recall <- dt_conf_matrix$byClass["Sensitivity"]        # Recall
dt_f1 <- 2 * (dt_precision * dt_recall) / (dt_precision + dt_recall)

# Print metrics from Decision Tree
cat("Decision Tree Metrics:\n")
cat("Precision:", round(dt_precision, 2), "\n")
cat("Recall:", round(dt_recall, 2), "\n")
cat("F1-Score:", round(dt_f1, 2), "\n\n")

# Extract metrics from Random Forest Confusion Matrix
rf_precision <- rf_conf_matrix$byClass["Pos Pred Value"]  # Precision
rf_recall <- rf_conf_matrix$byClass["Sensitivity"]        # Recall
rf_f1 <- 2 * (rf_precision * rf_recall) / (rf_precision + rf_recall)

# Print metrics from Random Forest
cat("Random Forest Metrics:\n")
cat("Precision:", round(rf_precision, 2), "\n")
cat("Recall:", round(rf_recall, 2), "\n")
cat("F1-Score:", round(rf_f1, 2), "\n")


# Interpretation of Metrics
   
# Precision: Measures how often the model's positive predictions (requiring ICU) are correct. A high precision minimizes false positives.
# Recall: Measures how well the model identifies actual positives (requiring ICU). A high recall minimizes false negatives.
# F1-Score: Combines precision and recall into a single metric, useful when there’s an uneven class distribution
# or when both false positives and false negatives are costly.


# Random Forest Model Tuning

# Tune Hyperparameters Using caret Package to help improve the model's performance.

# Summary of Tuning Process
# Grid Search: performe a grid search over mtry, ntree, and nodesize using 5-fold cross-validation.
# Best Hyperparameters: Identified the optimal combination of hyperparameters.
# Final Model: Used the optimal hyperparameters to train a final Random Forest model.
# Evaluation: Evaluated model performance on the test set using accuracy, precision, recall, F1-score, and confusion matrix.


# The caret package allows tuning using cross-validation. The following hyperparameters will be tunned:

# ntree: The number of trees in the forest.
# mtry: The number of variables randomly sampled as candidates at each split.
# nodesize: The minimum size of terminal nodes (leaf nodes).

# Define a grid of hyperparameters
tune_grid <- expand.grid(
  mtry = c(2, 3, 4, 5)       # Number of variables randomly sampled for each split
)

# Train Control
train_control <- trainControl(
  method = "cv",                # Cross-validation method
  number = 5,                   # 5-fold cross-validation
  classProbs = TRUE,            # Need probabilities for classification
  summaryFunction = twoClassSummary,  # Use metrics for classification
  verboseIter = TRUE            # Print progress during training
)


# Verify the levels
levels(balanced_train_data$confirmed_icu)

# Check for missing values in the training data
sum(is.na(balanced_train_data))
summary(balanced_train_data)

# Train the Random Forest model with the adjusted tuning grid
# The metric ROC (Receiver Operating Characteristic curve) is often more informative
# in imbalanced classification problems because it accounts for both the true positives and false positives, even when the classes are not balanced.
set.seed(324)
tuned_rf <- train(
  confirmed_icu ~ ., 
  data = balanced_train_data, 
  method = "rf", 
  trControl = train_control, 
  tuneGrid = tune_grid, 
  metric = "ROC", 
  importance = TRUE,    # Track feature importance
  ntree = 200,          # Set the number of trees (fixed)
  nodesize = 5          # Set the minimum node size (fixed)
)

# Print the best parameters
print(tuned_rf)

# Get the best hyperparameters
best_params <- tuned_rf$bestTune
print(best_params)

# Train the final Random Forest model with the best parameters
final_rf <- randomForest(
  confirmed_icu ~ ., 
  data = balanced_train_data, 
  ntree = 200, 
  mtry = best_params$mtry, 
  nodesize = 5
)

# Print final model summary
# print(final_rf)

# Predict on test data
predictions <- predict(final_rf, test_data)

# Evaluate the model with a confusion matrix
final_rf_conf_matrix <- confusionMatrix(predictions, test_data$confirmed_icu)
print(final_rf_conf_matrix)

# Extract additional metrics from Random Forest Confusion Matrix
final_rf_precision <- final_rf_conf_matrix$byClass["Pos Pred Value"]  # Precision
final_rf_recall <- final_rf_conf_matrix$byClass["Sensitivity"]        # Recall
final_rf_f1 <- 2 * (rf_precision * rf_recall) / (rf_precision + rf_recall)

# Print metrics for Tuned Random Forest model
cat("Tuned Random Forest Metrics:\n")
cat("Precision:", round(final_rf_precision, 2), "\n")
cat("Recall:", round(final_rf_recall, 2), "\n")
cat("F1-Score:", round(final_rf_f1, 2), "\n")

# same results... the tuning operation didn't improve the model's performance



##########################
# Other Models  

# Other models known for performing better on imbalanced datasets are:  XGBoost, LightGBM, and Gradient Boosting Machines (GBM).

# The GBM model will be utilized for predicting the ICU requirement, to see if it yields better results than the other two models built so far.

# **Gradient Boosting Machines (GBM)**  
  
#   Simplicity: GBM is simpler to implement compared to other models like XGBoost and LightGBM. It uses a standard boosting framework and is readily available in R through the caret package. The XGBoost and LightGBM models require additional libraries and more complex parameter tuning.  

# Flexibility with caret: GBM integrates seamlessly with the caret package, which simplifies hyperparameter tuning and cross-validation.
# The implementation in caret allows for straightforward handling of class imbalance using techniques like SMOTE or class weights.  



# Train a GBM model:

gbm_train_control <- trainControl(
  method = "cv",                # Cross-validation
  number = 5,                   # 5-fold cross-validation
  classProbs = TRUE,            # Enable class probabilities
  sampling = "smote",           # Use SMOTE for balancing classes
  summaryFunction = twoClassSummary,  # Evaluate using metrics like ROC
  verboseIter = TRUE            # Print progress during training
)

gbm_grid <- expand.grid(
  n.trees = c(50, 100, 150),         # Number of trees
  interaction.depth = c(1, 3, 5),    # Depth of each tree
  shrinkage = c(0.01, 0.1, 0.2),     # Learning rate
  n.minobsinnode = c(10, 20)         # Minimum samples in terminal nodes
)

set.seed(324)
tuned_gbm <- train(
  confirmed_icu ~ ., 
  data = balanced_train_data, 
  method = "gbm",                  # Specify GBM as the model
  trControl = gbm_train_control,   # Use the defined train control
  tuneGrid = gbm_grid,             # Use the tuning grid
  metric = "ROC",                  # Optimize based on ROC
  verbose = FALSE                  # Suppress verbose output from GBM
)

# Print the results of the tuning
print(tuned_gbm)

gbm_predictions <- predict(tuned_gbm, newdata = test_data)
gbm_conf_matrix <- confusionMatrix(gbm_predictions, test_data$confirmed_icu)
print(gbm_conf_matrix)

# Extract additional metrics from Random Forest Confusion Matrix
gbm_precision <- gbm_conf_matrix$byClass["Pos Pred Value"]  # Precision
gbm_recall <- gbm_conf_matrix$byClass["Sensitivity"]        # Recall
gbm_f1 <- 2 * (gbm_precision * rf_recall) / (gbm_precision + gbm_recall)

# Print metrics
cat("GBM Metrics:\n")
cat("Precision:", round(gbm_precision, 2), "\n")
cat("Recall:", round(gbm_recall, 2), "\n")
cat("F1-Score:", round(gbm_f1, 2), "\n")


# Unfortunately, the Gradient Boosting Machines (GBM) model didn't performed any better compared to the other model.
# The results are the same as the tuned Random Forest model.

 

