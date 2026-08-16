# Diabetes Data Analysis and Prediction

This repository contains an end-to-end data analysis and machine learning pipeline designed to predict the onset of diabetes based on diagnostic medical metrics. The project includes data cleaning, exploratory data analysis (EDA), handling class imbalance, and advanced hyperparameter tuning for classification models.

## Project Overview

The core of this project is the `Self data analysis.ipynb` notebook, which processes clinical data to build predictive models. The objective is to accurately classify whether a patient is diabetic (Outcome = 1) or healthy (Outcome = 0) using algorithms like Logistic Regression, Random Forest, and XGBoost.

## Dataset

The analysis utilizes a standard diabetes dataset (`diabetes.csv`) containing 768 patient records with the following diagnostic features:
*   `Pregnancies`: Number of times pregnant
*   `Glucose`: Plasma glucose concentration
*   `BloodPressure`: Diastolic blood pressure (mm Hg)
*   `SkinThickness`: Triceps skinfold thickness (mm)
*   `Insulin`: 2-Hour serum insulin (mu U/ml)
*   `BMI`: Body mass index (weight in kg/(height in m)^2)
*   `DiabetesPedigreeFunction`: Diabetes pedigree function
*   `Age`: Age (years)
*   `Outcome`: Class variable (0 or 1)

## Technologies Used

*   **Python 3.x**
*   **Data Manipulation & Math:** Pandas, NumPy
*   **Data Visualization:** Matplotlib, Seaborn
*   **Machine Learning:** Scikit-learn, XGBoost
*   **Data Balancing:** Imbalanced-learn (SMOTE)
*   **Hyperparameter Tuning:** Optuna

## Key Steps Performed

1.  **Data Cleaning & Imputation:** 
    *   Identified biologically impossible `0` values in clinical columns (Glucose, BloodPressure, SkinThickness, Insulin, BMI) and replaced them with `NaN`.
    *   Imputed the missing values using a **K-Nearest Neighbors (KNN) Imputer** (`n_neighbors=5`).
2.  **Exploratory Data Analysis (EDA):**
    *   Generated correlation heatmaps to identify relationships between clinical features.
    *   Visualized distributions using boxplots, histograms, and targeted pairplots to observe how features vary by diabetic outcome.
3.  **Data Preprocessing:**
    *   Split the data using a stratified train-test split to maintain the 2:1 ratio of healthy to diabetic patients.
    *   Applied **SMOTE (Synthetic Minority Over-sampling Technique)** strictly to the training data to balance the classes.
    *   Scaled all features using `StandardScaler` for optimal model performance.
4.  **Model Training & Baseline Evaluation:**
    *   Trained Logistic Regression, Random Forest, and XGBoost classifiers.
    *   Evaluated baseline models using ROC-AUC scores and classification reports, with XGBoost and Random Forest leading the initial leaderboard.
5.  **Advanced Hyperparameter Tuning:**
    *   Utilized **Optuna** to run 50 optimization trials for both the Random Forest and XGBoost models, maximizing the ROC-AUC score through cross-validation.
    *   Achieved highly optimized architectures (e.g., tuned Random Forest achieved an ROC-AUC of ~0.89 in cross-validation).
6.  **Model Insights:**
    *   Visualized performance using Confusion Matrices.
    *   Extracted **Feature Importance**, revealing that `Glucose`, `BMI`, and `Insulin` were the primary drivers for the model's predictions.

## How to Run

1.  Clone this repository:
    ```bash
    git clone [https://github.com/Shashwat-Singh536/Diabetes-Data-Analysis.git](https://github.com/Shashwat-Singh536/Diabetes-Data-Analysis.git)
    ```
2.  Navigate to the project directory:
    ```bash
    cd Diabetes-Data-Analysis
    ```
3.  Install the required dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn optuna
    ```
4.  Ensure `diabetes.csv` is in the root directory.
5.  Launch the Jupyter Notebook:
    ```bash
    jupyter notebook "Self data analysis.ipynb"
    ```
