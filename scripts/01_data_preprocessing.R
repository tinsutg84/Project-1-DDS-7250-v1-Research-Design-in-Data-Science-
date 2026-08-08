############################################################
# 01 - DATA PREPROCESSING
# Stroke Prediction Using Machine Learning
# DDS-7255 Reproducible Research Project
# CRISP-DM: Data Understanding and Data Preparation
############################################################

# ==========================================================
# 0. CLEAR ENVIRONMENT
# ==========================================================

rm(list = ls())
set.seed(123)

# ==========================================================
# 1. INSTALL AND LOAD PACKAGES
# ==========================================================

packages <- c(
  "tidyverse",
  "caret",
  "recipes",
  "themis",
  "randomForest",
  "pROC",
  "fmsb",
  "patchwork",
  "knitr",
  "scales"
)

installed <- packages %in% rownames(installed.packages())

if (any(!installed)) {
  install.packages(packages[!installed])
}

library(tidyverse)
library(caret)
library(recipes)
library(themis)
library(randomForest)
library(fmsb)
library(patchwork)
library(knitr)

# ==========================================================
# 2. IMPORT DATA
# ==========================================================

# Relative path makes the project reproducible on other computers.
file_path <- "data/healthcare-dataset-stroke-data.csv"

stroke_raw <- read.csv(
  file_path,
  stringsAsFactors = FALSE,
  na.strings = c("N/A", "NA", "")
)

# Ensure BMI is numeric.
stroke_raw$bmi <- as.numeric(stroke_raw$bmi)

cat("\nOriginal dataset dimensions:\n")
print(dim(stroke_raw))

cat("\nOriginal dataset structure:\n")
str(stroke_raw)

cat("\nMissing values by variable:\n")
print(colSums(is.na(stroke_raw)))

cat("\nGender distribution before cleaning:\n")
print(table(stroke_raw$gender, useNA = "ifany"))

# ==========================================================
# 3. DATA CLEANING
# ==========================================================

# Remove ID before EDA, feature selection, and modeling.
stroke_clean <- stroke_raw %>%
  select(-id)

# Count the rare gender category before removing it.
rare_gender_count <- sum(
  stroke_clean$gender == "Other",
  na.rm = TRUE
)

cat("\nNumber of records in the rare gender category:\n")
print(rare_gender_count)

# Remove the single sparse category.
stroke_clean <- stroke_clean %>%
  filter(gender != "Other")

# Convert categorical variables to factors.
stroke_clean <- stroke_clean %>%
  mutate(
    stroke = factor(
      stroke,
      levels = c(0, 1),
      labels = c("No", "Yes")
    ),
    hypertension = factor(
      hypertension,
      levels = c(0, 1),
      labels = c("No", "Yes")
    ),
    heart_disease = factor(
      heart_disease,
      levels = c(0, 1),
      labels = c("No", "Yes")
    ),
    gender = factor(gender),
    ever_married = factor(ever_married),
    work_type = factor(work_type),
    Residence_type = factor(Residence_type),
    smoking_status = factor(smoking_status)
  )

cat("\nDataset dimensions after cleaning:\n")
print(dim(stroke_clean))

cat("\nCleaned dataset structure:\n")
str(stroke_clean)
