############################################################
# 02 - EXPLORATORY DATA ANALYSIS AND FEATURE ENGINEERING
# Stroke Prediction Using Machine Learning
# DDS-7255 Reproducible Research Project
# CRISP-DM: Data Understanding and Data Preparation
############################################################

# This script assumes 01_data_preprocessing.R has been run.

# ==========================================================
# 4. EXPLORATORY DATA ANALYSIS
# ==========================================================

cat("\nDataset summary:\n")
print(summary(stroke_clean))

cat("\nStroke distribution:\n")
print(table(stroke_clean$stroke))

cat("\nStroke proportions:\n")
print(round(prop.table(table(stroke_clean$stroke)), 4))

# ==========================================================
# 4A. CREATE DOMAIN-BASED GROUPS FOR EDA
# ==========================================================

stroke_eda <- stroke_clean %>%
  mutate(
    age_group = case_when(
      age < 40 ~ "Under 40",
      age >= 40 & age < 60 ~ "40–59",
      age >= 60 ~ "60 or Older"
    ),

    bmi_category = case_when(
      is.na(bmi) ~ "Missing",
      bmi < 18.5 ~ "Underweight",
      bmi >= 18.5 & bmi < 25 ~ "Normal",
      bmi >= 25 & bmi < 30 ~ "Overweight",
      bmi >= 30 ~ "Obese"
    ),

    glucose_category = case_when(
      avg_glucose_level < 100 ~ "Below 100",
      avg_glucose_level >= 100 &
        avg_glucose_level < 126 ~ "100–125",
      avg_glucose_level >= 126 ~ "126 or Higher"
    ),

    cardiovascular_risk_count =
      ifelse(hypertension == "Yes", 1, 0) +
      ifelse(heart_disease == "Yes", 1, 0)
  )

# ==========================================================
# 4B. EDA SUMMARY TABLES
# ==========================================================

age_stroke_summary <- stroke_eda %>%
  group_by(age_group) %>%
  summarise(
    Patients = n(),
    Stroke_Cases = sum(stroke == "Yes"),
    Stroke_Rate = mean(stroke == "Yes"),
    .groups = "drop"
  )

glucose_stroke_summary <- stroke_eda %>%
  group_by(glucose_category) %>%
  summarise(
    Patients = n(),
    Stroke_Cases = sum(stroke == "Yes"),
    Stroke_Rate = mean(stroke == "Yes"),
    .groups = "drop"
  )

bmi_stroke_summary <- stroke_eda %>%
  group_by(bmi_category) %>%
  summarise(
    Patients = n(),
    Stroke_Cases = sum(stroke == "Yes"),
    Stroke_Rate = mean(stroke == "Yes"),
    .groups = "drop"
  )

cardio_stroke_summary <- stroke_eda %>%
  group_by(cardiovascular_risk_count) %>%
  summarise(
    Patients = n(),
    Stroke_Cases = sum(stroke == "Yes"),
    Stroke_Rate = mean(stroke == "Yes"),
    .groups = "drop"
  )

print(age_stroke_summary)
print(glucose_stroke_summary)
print(bmi_stroke_summary)
print(cardio_stroke_summary)

# ==========================================================
# 4C. EDA VISUALIZATIONS
# ==========================================================

plot_class <- ggplot(stroke_eda, aes(x = stroke)) +
  geom_bar() +
  labs(
    title = "Stroke Class Distribution",
    x = "Stroke Status",
    y = "Number of Patients"
  ) +
  theme_minimal()

plot_age <- ggplot(
  age_stroke_summary,
  aes(x = age_group, y = Stroke_Rate)
) +
  geom_col() +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Stroke Rate by Age Group",
    x = "Age Group",
    y = "Stroke Rate"
  ) +
  theme_minimal()

plot_glucose <- ggplot(
  glucose_stroke_summary,
  aes(x = glucose_category, y = Stroke_Rate)
) +
  geom_col() +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Stroke Rate by Glucose Category",
    x = "Average Glucose Category",
    y = "Stroke Rate"
  ) +
  theme_minimal()

plot_bmi <- ggplot(
  bmi_stroke_summary,
  aes(
    x = reorder(bmi_category, Stroke_Rate),
    y = Stroke_Rate
  )
) +
  geom_col() +
  coord_flip() +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Stroke Rate by BMI Category",
    x = "BMI Category",
    y = "Stroke Rate"
  ) +
  theme_minimal()

plot_cardio <- ggplot(
  cardio_stroke_summary,
  aes(
    x = factor(cardiovascular_risk_count),
    y = Stroke_Rate
  )
) +
  geom_col() +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Stroke Rate by Cardiovascular Risk Count",
    x = "Number of Cardiovascular Conditions",
    y = "Stroke Rate"
  ) +
  theme_minimal()

print(plot_class)

eda_combined <- (
  plot_age + plot_glucose
) / (
  plot_bmi + plot_cardio
)

print(eda_combined)

# ==========================================================
# 5. FEATURE CREATION
# ==========================================================

stroke_model <- stroke_clean %>%
  mutate(
    age_group = factor(
      case_when(
        age < 40 ~ "Under_40",
        age >= 40 & age < 60 ~ "Age_40_59",
        age >= 60 ~ "Age_60_Plus"
      )
    ),

    bmi_category = factor(
      case_when(
        is.na(bmi) ~ NA_character_,
        bmi < 18.5 ~ "Underweight",
        bmi >= 18.5 & bmi < 25 ~ "Normal",
        bmi >= 25 & bmi < 30 ~ "Overweight",
        bmi >= 30 ~ "Obese"
      )
    ),

    glucose_category = factor(
      case_when(
        avg_glucose_level < 100 ~ "Below_100",
        avg_glucose_level >= 100 &
          avg_glucose_level < 126 ~ "Range_100_125",
        avg_glucose_level >= 126 ~ "Range_126_Plus"
      )
    ),

    hypertension_num =
      ifelse(hypertension == "Yes", 1, 0),

    heart_disease_num =
      ifelse(heart_disease == "Yes", 1, 0),

    cardiovascular_risk_count =
      hypertension_num + heart_disease_num,

    older_high_glucose =
      ifelse(
        age >= 60 &
          avg_glucose_level >= 126,
        1,
        0
      ),

    bmi_glucose_interaction =
      bmi * avg_glucose_level,

    age_glucose_interaction =
      age * avg_glucose_level,

    age_squared = age^2
  )

cat("\nFeature engineering completed.\n")
cat("Modeling dataset dimensions:\n")
print(dim(stroke_model))
