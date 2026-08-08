############################################################
# 04 - FEATURE SELECTION AND MODEL DEVELOPMENT
# Stroke Prediction Using Machine Learning
# DDS-7255 Reproducible Research Project
# CRISP-DM: Modeling
############################################################

# This script assumes scripts 01, 02, and 03
# have already been run in the same R session.

# ==========================================================
# 9. METHOD 1: RANDOM FOREST FEATURE IMPORTANCE
# ==========================================================

set.seed(123)

rf_importance_model <- randomForest(
  stroke ~ .,
  data = train_balanced,
  ntree = 500,
  mtry = 3,
  nodesize = 5,
  importance = TRUE
)

rf_importance_values <- importance(
  rf_importance_model
)

rf_importance_table <- data.frame(
  Feature = rownames(rf_importance_values),
  MeanDecreaseGini =
    rf_importance_values[, "MeanDecreaseGini"],
  row.names = NULL
) %>%
  arrange(desc(MeanDecreaseGini))

cat("\nTop 20 Random Forest importance features:\n")
print(head(rf_importance_table, 20))

# Select top 10 features for baseline methodology.
rf_selected_features <- rf_importance_table %>%
  slice_head(n = 10) %>%
  pull(Feature)

cat("\nFeatures selected using Random Forest importance:\n")
print(rf_selected_features)

rf_importance_plot <- rf_importance_table %>%
  slice_head(n = 15) %>%
  ggplot(
    aes(
      x = reorder(
        Feature,
        MeanDecreaseGini
      ),
      y = MeanDecreaseGini
    )
  ) +
  geom_col() +
  coord_flip() +
  labs(
    title =
      "Top Features Selected by Random Forest Importance",
    x = "Feature",
    y = "Mean Decrease in Gini"
  ) +
  theme_minimal()

print(rf_importance_plot)

# ==========================================================
# 10. METHOD 2: RECURSIVE FEATURE ELIMINATION
# ==========================================================

x_train_rfe <- train_balanced %>%
  select(-stroke)

y_train_rfe <- train_balanced$stroke

candidate_sizes <- c(
  5,
  8,
  10,
  12,
  15,
  20
)

candidate_sizes <- candidate_sizes[
  candidate_sizes <= ncol(x_train_rfe)
]

# ----------------------------------------------------------
# CUSTOM F1 SCORING FUNCTION
# ----------------------------------------------------------
# Accuracy is not used as the RFE optimization criterion
# because the stroke outcome is highly imbalanced.

rfFuncs_F1 <- rfFuncs

rfFuncs_F1$summary <- function(
  data,
  lev = NULL,
  model = NULL
) {

  precision_value <- posPredValue(
    data$pred,
    data$obs,
    positive = "Yes"
  )

  recall_value <- sensitivity(
    data$pred,
    data$obs,
    positive = "Yes"
  )

  if (
    is.na(precision_value) ||
    is.na(recall_value) ||
    (precision_value + recall_value) == 0
  ) {
    f1_value <- 0
  } else {
    f1_value <-
      2 *
      precision_value *
      recall_value /
      (precision_value + recall_value)
  }

  c(
    F1 = as.numeric(f1_value)
  )
}

rfe_control <- rfeControl(
  functions = rfFuncs_F1,
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  verbose = FALSE,
  returnResamp = "final",
  allowParallel = TRUE
)

set.seed(123)

rfe_results <- rfe(
  x = x_train_rfe,
  y = y_train_rfe,
  sizes = candidate_sizes,
  rfeControl = rfe_control,
  metric = "F1",
  maximize = TRUE,
  ntree = 500
)

cat("\nRFE summary:\n")
print(rfe_results)

cat("\nRFE performance by subset size:\n")
print(rfe_results$results)

# Restrict comparison to the candidate sizes requested.
rfe_candidate_results <- rfe_results$results %>%
  filter(
    Variables %in% candidate_sizes
  )

# Select feature subset with highest cross-validated F1.
rfe_best_size <- rfe_candidate_results %>%
  arrange(
    desc(F1)
  ) %>%
  slice(1) %>%
  pull(Variables)

cat("\nBest RFE subset size based on F1-score:\n")
print(rfe_best_size)

# ==========================================================
# RFE VARIABLE RANKING
# ==========================================================

rfe_variable_ranking <- rfe_results$variables %>%
  filter(
    Variables == rfe_best_size
  ) %>%
  group_by(var) %>%
  summarise(
    Overall = mean(
      Overall,
      na.rm = TRUE
    ),
    .groups = "drop"
  ) %>%
  arrange(
    desc(Overall)
  )

rfe_selected_features <- rfe_variable_ranking %>%
  slice_head(
    n = rfe_best_size
  ) %>%
  pull(var)

cat("\nFeatures selected using RFE:\n")
print(rfe_selected_features)

# ==========================================================
# RFE PERFORMANCE PLOT
# ==========================================================

rfe_performance_plot <- ggplot(
  rfe_candidate_results,
  aes(
    x = Variables,
    y = F1
  )
) +
  geom_line() +
  geom_point() +
  labs(
    title =
      "RFE Performance by Number of Selected Features",
    x = "Number of Features",
    y = "Cross-Validated F1 Score"
  ) +
  theme_minimal()

print(rfe_performance_plot)

# ==========================================================
# 11. PREPARE FEATURE-SPECIFIC DATASETS
# ==========================================================

rf_train <- train_balanced %>%
  select(
    all_of(rf_selected_features),
    stroke
  )

rf_validation <- validation_processed %>%
  select(
    all_of(rf_selected_features),
    stroke
  )

rf_test <- test_processed %>%
  select(
    all_of(rf_selected_features),
    stroke
  )

rfe_train <- train_balanced %>%
  select(
    all_of(rfe_selected_features),
    stroke
  )

rfe_validation <- validation_processed %>%
  select(
    all_of(rfe_selected_features),
    stroke
  )

rfe_test <- test_processed %>%
  select(
    all_of(rfe_selected_features),
    stroke
  )

# ==========================================================
# 12. TRAIN TWO RANDOM FOREST MODELS
# ==========================================================

# Identical model settings are used for a fair comparison.

set.seed(123)

rf_feature_model <- randomForest(
  stroke ~ .,
  data = rf_train,
  ntree = 500,
  mtry = 3,
  nodesize = 5,
  importance = TRUE
)

set.seed(123)

rfe_feature_model <- randomForest(
  stroke ~ .,
  data = rfe_train,
  ntree = 500,
  mtry = 3,
  nodesize = 5,
  importance = TRUE
)

cat("\nRandom Forest models successfully trained.\n")

cat("\nRF Importance model number of features:\n")
print(length(rf_selected_features))

cat("\nRFE model number of features:\n")
print(length(rfe_selected_features))

# ==========================================================
# FEATURE-SELECTION SUMMARY
# ==========================================================

feature_overlap <- intersect(
  rf_selected_features,
  rfe_selected_features
)

rf_only_features <- setdiff(
  rf_selected_features,
  rfe_selected_features
)

rfe_only_features <- setdiff(
  rfe_selected_features,
  rf_selected_features
)

cat("\nFeatures selected by both methods:\n")
print(feature_overlap)

cat("\nFeatures selected only by RF importance:\n")
print(rf_only_features)

cat("\nFeatures selected only by RFE:\n")
print(rfe_only_features)

selection_summary <- tibble(
  Method = c(
    "Random Forest Importance",
    "Recursive Feature Elimination"
  ),
  Number_of_Features = c(
    length(rf_selected_features),
    length(rfe_selected_features)
  ),
  Selected_Features = c(
    paste(
      rf_selected_features,
      collapse = ", "
    ),
    paste(
      rfe_selected_features,
      collapse = ", "
    )
  )
)

print(selection_summary)

cat(
  "\nFeature selection and model development complete.\n"
)
