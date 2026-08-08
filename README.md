# Stroke Prediction Using Machine Learning

## Project Overview

This project applies reproducible data science practices to stroke prediction using machine learning. The analysis examines demographic, behavioral, and clinical risk factors associated with stroke and compares two feature-selection approaches: Random Forest feature importance and Recursive Feature Elimination (RFE).

The project follows the Cross-Industry Standard Process for Data Mining (CRISP-DM) and organizes the analytical workflow into separate modules to improve transparency, reproducibility, and maintainability.

## Dataset Information

The project uses the Healthcare Stroke Prediction Dataset obtained from Kaggle. The dataset contains demographic, behavioral, and clinical variables including age, gender, hypertension, heart disease, average glucose level, BMI, smoking status, residence type, and stroke status.

The outcome variable is `stroke`, where:

- 0 = No stroke
- 1 = Stroke

The dataset is highly imbalanced, with fewer than 5% of observations representing stroke cases. Therefore, model evaluation emphasizes recall, F1-score, balanced accuracy, and ROC-AUC rather than overall accuracy alone.

The dataset is located in:

`data/healthcare-dataset-stroke-data.csv`

## CRISP-DM Process

The project follows the six stages of CRISP-DM.

### 1. Business Understanding
The primary objective is to investigate whether feature engineering and feature-selection techniques can improve machine learning identification of patients at elevated risk of stroke.

### 2. Data Understanding
Exploratory data analysis was conducted to examine class imbalance, missing values, variable distributions, and relationships between stroke and major clinical risk factors.

### 3. Data Preparation
Data preparation included:

- Removal of unnecessary identifiers
- Training and testing partitioning
- Training-based median imputation for missing BMI values
- Categorical variable encoding
- Feature engineering
- Interaction-term creation
- SMOTE applied only to training data

These procedures were designed to reduce information leakage and improve reproducibility.

### 4. Modeling
Random Forest was used as the primary classification algorithm. Two feature-selection approaches were compared:

1. Random Forest feature importance
2. Recursive Feature Elimination (RFE)

Both approaches used consistent preprocessing procedures and modeling parameters to support a fair comparison.

### 5. Evaluation
Model performance was evaluated using multiple metrics appropriate for imbalanced classification, including:

- Recall
- Precision
- F1-score
- Balanced accuracy
- ROC-AUC
- Confusion matrices

Classification thresholds were also evaluated to improve identification of the minority stroke class.

### 6. Deployment
The project is organized as a public reproducible research repository. Code, data, documentation, and analytical outputs are provided so that the workflow can be reviewed and reproduced.

## Repository Structure

    ├── data/
    │   ├── healthcare-dataset-stroke-data.csv
    │   └── README.md
    │
    ├── scripts/
    │   ├── 01_data_preprocessing.R
    │   ├── 02_eda_feature_engineering.R
    │   ├── 03_split_preprocessing_smote.R
    │   ├── 04_feature_selection_modeling.R
    │   └── 05_threshold_evaluation_visualization.R
    │
    ├── notebooks/
    │   ├── stroke_prediction_analysis.Rmd
    │   └── README.md
    │
    ├── outputs/
    │   ├── README.md
    │   └── project visualizations
    │
    ├── requirements.txt
    └── README.md

## How to Run the Project

### Step 1: Download or Clone the Repository

Download the repository from GitHub or clone it to your local computer.

### Step 2: Install R and RStudio

Install a current version of R and, preferably, RStudio.

### Step 3: Install Required Packages

Install the R packages listed in `requirements.txt`.

For example:

    install.packages(c(
      "tidyverse",
      "caret",
      "randomForest",
      "pROC",
      "ggplot2",
      "dplyr",
      "tidyr",
      "recipes",
      "themis",
      "rmarkdown",
      "knitr"
    ))

### Step 4: Run the Scripts

Run the scripts in numerical order:

    01_data_preprocessing.R
    02_eda_feature_engineering.R
    03_split_preprocessing_smote.R
    04_feature_selection_modeling.R
    05_threshold_evaluation_visualization.R

Alternatively, open the complete analysis:

    notebooks/stroke_prediction_analysis.Rmd

and run or knit the R Markdown document.

## Dependencies

The project was developed using R. Major packages include:

- tidyverse
- caret
- randomForest
- pROC
- ggplot2
- dplyr
- tidyr
- recipes
- themis
- rmarkdown
- knitr

See `requirements.txt` for dependency information.

## Results and Insights

Exploratory analysis demonstrated substantial class imbalance and important relationships between stroke and several clinical and demographic characteristics. Stroke occurrence was notably associated with older age, elevated glucose levels, and cardiovascular risk.

Feature engineering generated additional predictors including age groups, BMI categories, glucose categories, cardiovascular risk counts, squared age, and interaction terms.

Random Forest feature importance identified age-related variables, age squared, age-glucose interaction, BMI, and average glucose level among important predictors.

RFE was also evaluated as an alternative feature-selection strategy. Model evaluation emphasized minority-class detection rather than overall accuracy because high accuracy can be misleading in a highly imbalanced dataset.

The results demonstrate the importance of combining appropriate preprocessing, feature engineering, feature selection, threshold evaluation, and clinically meaningful performance metrics when developing machine learning models for healthcare applications.

## Reproducibility

To improve reproducibility:

- The analytical workflow is divided into modular scripts.
- The raw dataset is included in the repository.
- Preprocessing and feature engineering procedures are documented.
- Training-derived preprocessing values are applied to test data to prevent information leakage.
- SMOTE is restricted to training data.
- Model evaluation procedures are documented.
- Visual outputs are retained in the `outputs/` directory.
- Dependencies are documented in `requirements.txt`.

## Author

Tinsae Abdeta

DDS-7250 – Research Design in Data Science

National University
