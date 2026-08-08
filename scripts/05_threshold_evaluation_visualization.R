############################################################
# 05 - THRESHOLD SELECTION, MODEL EVALUATION, AND VISUALIZATION
# Stroke Prediction Using Machine Learning
# DDS-7255 Reproducible Research Project
# CRISP-DM: Evaluation
############################################################

# This script assumes scripts 01 through 04
# have already been run in the same R session.

# ==========================================================
# 13. SELECT CLASSIFICATION THRESHOLDS
# ==========================================================

rf_validation_probability <- predict(
  rf_feature_model,
  newdata = rf_validation,
  type = "prob"
)[, "Yes"]

rfe_validation_probability <- predict(
  rfe_feature_model,
  newdata = rfe_validation,
  type = "prob"
)[, "Yes"]

calculate_threshold_metrics <- function(
  actual,
  probability,
  threshold
) {

  prediction <- factor(
    ifelse(
      probability >= threshold,
      "Yes",
      "No"
    ),
    levels = c("No", "Yes")
  )

  cm <- confusionMatrix(
    prediction,
    actual,
    positive = "Yes"
  )

  data.frame(
    Threshold = threshold,
    Precision =
      as.numeric(cm$byClass["Precision"]),
    Recall =
      as.numeric(cm$byClass["Sensitivity"]),
    F1 =
      as.numeric(cm$byClass["F1"]),
    Balanced_Accuracy =
      as.numeric(cm$byClass["Balanced Accuracy"])
  )
}

threshold_grid <- seq(
  from = 0.05,
  to = 0.95,
  by = 0.01
)

rf_threshold_results <- bind_rows(
  lapply(
    threshold_grid,
    function(x) {
      calculate_threshold_metrics(
        actual = rf_validation$stroke,
        probability = rf_validation_probability,
        threshold = x
      )
    }
  )
)

rfe_threshold_results <- bind_rows(
  lapply(
    threshold_grid,
    function(x) {
      calculate_threshold_metrics(
        actual = rfe_validation$stroke,
        probability = rfe_validation_probability,
        threshold = x
      )
    }
  )
)

rf_threshold_results <- rf_threshold_results %>%
  mutate(
    Precision = replace_na(Precision, 0),
    Recall = replace_na(Recall, 0),
    F1 = replace_na(F1, 0),
    Balanced_Accuracy =
      replace_na(Balanced_Accuracy, 0)
  )

rfe_threshold_results <- rfe_threshold_results %>%
  mutate(
    Precision = replace_na(Precision, 0),
    Recall = replace_na(Recall, 0),
    F1 = replace_na(F1, 0),
    Balanced_Accuracy =
      replace_na(Balanced_Accuracy, 0)
  )

# Select thresholds using validation F1-score.
rf_best_threshold <- rf_threshold_results %>%
  arrange(
    desc(F1),
    desc(Balanced_Accuracy)
  ) %>%
  slice(1) %>%
  pull(Threshold)

rfe_best_threshold <- rfe_threshold_results %>%
  arrange(
    desc(F1),
    desc(Balanced_Accuracy)
  ) %>%
  slice(1) %>%
  pull(Threshold)

cat("\nBest threshold for RF-importance model:\n")
print(rf_best_threshold)

cat("\nBest threshold for RFE model:\n")
print(rfe_best_threshold)

best_threshold_table <- tibble(
  Model = c(
    "Random Forest Importance",
    "Recursive Feature Elimination"
  ),
  Selected_Threshold = c(
    rf_best_threshold,
    rfe_best_threshold
  )
)

print(best_threshold_table)

# ==========================================================
# THRESHOLD VISUALIZATION
# ==========================================================

threshold_plot_data <- bind_rows(
  rf_threshold_results %>%
    mutate(Model = "RF Importance"),
  rfe_threshold_results %>%
    mutate(Model = "RFE")
)

threshold_plot <- ggplot(
  threshold_plot_data,
  aes(
    x = Threshold,
    y = F1,
    linetype = Model
  )
) +
  geom_line() +
  labs(
    title =
      "Validation F1 Score Across Classification Thresholds",
    x = "Probability Threshold",
    y = "F1 Score"
  ) +
  theme_minimal()

print(threshold_plot)

# ==========================================================
# 14. FINAL TEST SET PREDICTIONS
# ==========================================================

rf_test_probability <- predict(
  rf_feature_model,
  newdata = rf_test,
  type = "prob"
)[, "Yes"]

rfe_test_probability <- predict(
  rfe_feature_model,
  newdata = rfe_test,
  type = "prob"
)[, "Yes"]

rf_test_prediction <- factor(
  ifelse(
    rf_test_probability >= rf_best_threshold,
    "Yes",
    "No"
  ),
  levels = c("No", "Yes")
)

rfe_test_prediction <- factor(
  ifelse(
    rfe_test_probability >= rfe_best_threshold,
    "Yes",
    "No"
  ),
  levels = c("No", "Yes")
)

# ==========================================================
# 15. MODEL EVALUATION FUNCTION
# ==========================================================

evaluate_classifier <- function(
  actual,
  prediction,
  probability,
  model_name,
  threshold
) {

  cm <- confusionMatrix(
    prediction,
    actual,
    positive = "Yes"
  )

  roc_object <- pROC::roc(
    response = actual,
    predictor = probability,
    levels = c("No", "Yes"),
    direction = "<",
    quiet = TRUE
  )

  metrics <- data.frame(
    Model = model_name,
    Threshold = threshold,
    Accuracy =
      as.numeric(cm$overall["Accuracy"]),
    Precision =
      as.numeric(cm$byClass["Precision"]),
    Recall =
      as.numeric(cm$byClass["Sensitivity"]),
    Specificity =
      as.numeric(cm$byClass["Specificity"]),
    F1 =
      as.numeric(cm$byClass["F1"]),
    Balanced_Accuracy =
      as.numeric(cm$byClass["Balanced Accuracy"]),
    ROC_AUC =
      as.numeric(pROC::auc(roc_object))
  )

  list(
    metrics = metrics,
    confusion_matrix = cm,
    roc = roc_object
  )
}

rf_evaluation <- evaluate_classifier(
  actual = rf_test$stroke,
  prediction = rf_test_prediction,
  probability = rf_test_probability,
  model_name = "Random Forest Importance",
  threshold = rf_best_threshold
)

rfe_evaluation <- evaluate_classifier(
  actual = rfe_test$stroke,
  prediction = rfe_test_prediction,
  probability = rfe_test_probability,
  model_name = "Recursive Feature Elimination",
  threshold = rfe_best_threshold
)

# ==========================================================
# 16. MODEL COMPARISON TABLE
# ==========================================================

comparison_table <- bind_rows(
  rf_evaluation$metrics,
  rfe_evaluation$metrics
) %>%
  mutate(
    across(
      where(is.numeric),
      ~ round(.x, 4)
    )
  )

cat("\nFinal model comparison:\n")
print(comparison_table)

# ==========================================================
# 17. CONFUSION MATRICES
# ==========================================================

cat("\nRandom Forest Importance Confusion Matrix:\n")
print(rf_evaluation$confusion_matrix)

cat("\nRFE Confusion Matrix:\n")
print(rfe_evaluation$confusion_matrix)

plot_confusion_matrix <- function(
  actual,
  prediction,
  title
) {

  cm_data <- as.data.frame(
    table(
      Actual = actual,
      Predicted = prediction
    )
  )

  ggplot(
    cm_data,
    aes(
      x = Predicted,
      y = Actual,
      fill = Freq
    )
  ) +
    geom_tile() +
    geom_text(
      aes(label = Freq),
      size = 6
    ) +
    labs(
      title = title,
      x = "Predicted Class",
      y = "Actual Class"
    ) +
    theme_minimal() +
    theme(
      legend.position = "none"
    )
}

rf_confusion_plot <- plot_confusion_matrix(
  actual = rf_test$stroke,
  prediction = rf_test_prediction,
  title =
    "Confusion Matrix: Random Forest Importance"
)

rfe_confusion_plot <- plot_confusion_matrix(
  actual = rfe_test$stroke,
  prediction = rfe_test_prediction,
  title =
    "Confusion Matrix: Recursive Feature Elimination"
)

confusion_combined <- (
  rf_confusion_plot +
    rfe_confusion_plot
)

print(confusion_combined)

# ==========================================================
# 18. ROC CURVES
# ==========================================================

roc_list <- list(
  "Random Forest Importance" =
    rf_evaluation$roc,
  "Recursive Feature Elimination" =
    rfe_evaluation$roc
)

roc_plot <- pROC::ggroc(
  roc_list,
  legacy.axes = TRUE
) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed"
  ) +
  labs(
    title = "ROC Curve Comparison",
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_minimal()

print(roc_plot)

# ==========================================================
# 19. SPIDER / RADAR CHART
# ==========================================================

radar_metrics <- comparison_table %>%
  select(
    Model,
    Accuracy,
    Precision,
    Recall,
    F1,
    ROC_AUC
  )

radar_metrics[is.na(radar_metrics)] <- 0

radar_data <- radar_metrics %>%
  column_to_rownames("Model")

radar_data <- rbind(
  Maximum = rep(
    1,
    ncol(radar_data)
  ),
  Minimum = rep(
    0,
    ncol(radar_data)
  ),
  radar_data
)

fmsb::radarchart(
  radar_data,
  axistype = 1,
  pcol = 1:nrow(comparison_table),
  plwd = 3,
  plty = 1,
  cglty = 1,
  axislabcol = "black",
  caxislabels = seq(
    0,
    1,
    0.2
  ),
  cglwd = 0.8,
  vlcex = 1.1,
  title = "Model Performance Comparison"
)

legend(
  "topright",
  legend = comparison_table$Model,
  col = 1:nrow(comparison_table),
  lty = 1,
  lwd = 3,
  bty = "n"
)

# ==========================================================
# 20. FEATURE-SELECTION COMPARISON
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

# ==========================================================
# 21. PERFORMANCE METRIC BAR CHART
# ==========================================================

metric_long <- comparison_table %>%
  select(
    Model,
    Accuracy,
    Precision,
    Recall,
    F1,
    ROC_AUC
  ) %>%
  pivot_longer(
    cols = -Model,
    names_to = "Metric",
    values_to = "Value"
  )

metric_comparison_plot <- ggplot(
  metric_long,
  aes(
    x = Metric,
    y = Value,
    fill = Model
  )
) +
  geom_col(
    position = "dodge"
  ) +
  labs(
    title = "Comparison of Model Performance Metrics",
    x = "Performance Metric",
    y = "Score"
  ) +
  scale_y_continuous(
    limits = c(0, 1)
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    )
  )

print(metric_comparison_plot)

cat(
  "\nModel evaluation and visualization complete.\n"
)
