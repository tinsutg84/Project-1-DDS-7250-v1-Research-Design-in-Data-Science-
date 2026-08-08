############################################################
# 03 - DATA SPLIT, PREPROCESSING, AND SMOTE
# Stroke Prediction Using Machine Learning
# DDS-7255 Reproducible Research Project
# CRISP-DM: Data Preparation
############################################################

# This script assumes:
# 01_data_preprocessing.R
# 02_eda_feature_engineering.R
# have already been run.

# ==========================================================
# 6. STRATIFIED TRAIN, VALIDATION, AND TEST SPLIT
# ==========================================================

# 70% training, 15% validation, and 15% testing.

set.seed(123)

train_index <- createDataPartition(
  stroke_model$stroke,
  p = 0.70,
  list = FALSE
)

train_data <- stroke_model[
  train_index,
]

remaining_data <- stroke_model[
  -train_index,
]

set.seed(123)

validation_index <- createDataPartition(
  remaining_data$stroke,
  p = 0.50,
  list = FALSE
)

validation_data <- remaining_data[
  validation_index,
]

test_data <- remaining_data[
  -validation_index,
]

cat("\nTraining dimensions:\n")
print(dim(train_data))

cat("\nValidation dimensions:\n")
print(dim(validation_data))

cat("\nTesting dimensions:\n")
print(dim(test_data))

cat("\nTraining class distribution:\n")
print(table(train_data$stroke))

cat("\nValidation class distribution:\n")
print(table(validation_data$stroke))

cat("\nTesting class distribution:\n")
print(table(test_data$stroke))

# ==========================================================
# 7. PREPROCESSING
# ==========================================================

# IMPORTANT:
# All preprocessing parameters are learned from the
# training data only, then applied unchanged to
# validation and test data to prevent information leakage.

preprocess_recipe <- recipe(
  stroke ~ .,
  data = train_data
) %>%
  step_impute_median(
    bmi,
    bmi_glucose_interaction
  ) %>%
  step_impute_mode(
    bmi_category
  ) %>%
  step_unknown(
    all_nominal_predictors()
  ) %>%
  step_novel(
    all_nominal_predictors()
  ) %>%
  step_dummy(
    all_nominal_predictors(),
    one_hot = TRUE
  ) %>%
  step_zv(
    all_predictors()
  ) %>%
  step_normalize(
    all_numeric_predictors()
  )

prepared_recipe <- prep(
  preprocess_recipe,
  training = train_data,
  retain = TRUE
)

train_processed <- juice(
  prepared_recipe
)

validation_processed <- bake(
  prepared_recipe,
  new_data = validation_data
)

test_processed <- bake(
  prepared_recipe,
  new_data = test_data
)

cat("\nProcessed training dimensions:\n")
print(dim(train_processed))

cat("\nProcessed validation dimensions:\n")
print(dim(validation_processed))

cat("\nProcessed testing dimensions:\n")
print(dim(test_processed))

# ==========================================================
# 8. APPLY SMOTE TO TRAINING DATA ONLY
# ==========================================================

# SMOTE is applied only to the processed training data.
# Validation and test data remain untouched.

smote_recipe <- recipe(
  stroke ~ .,
  data = train_processed
) %>%
  step_smote(
    stroke,
    over_ratio = 1
  )

prepared_smote <- prep(
  smote_recipe,
  training = train_processed,
  retain = TRUE
)

train_balanced <- juice(
  prepared_smote
)

cat("\nTraining distribution before SMOTE:\n")
print(table(train_processed$stroke))

cat("\nTraining distribution after SMOTE:\n")
print(table(train_balanced$stroke))

cat("\nValidation distribution remains unchanged:\n")
print(table(validation_processed$stroke))

cat("\nTesting distribution remains unchanged:\n")
print(table(test_processed$stroke))

cat("\nData splitting, preprocessing, and SMOTE complete.\n")
