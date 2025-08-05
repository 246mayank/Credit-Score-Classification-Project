# Credit Score Classification Project

## Project Overview

This project is an end-to-end machine learning pipeline designed to classify customer credit scores into three categories: 'Good', 'Standard', and 'Poor'. The model was built using a dataset from Kaggle and demonstrates a complete workflow, including data cleaning, feature engineering, and model evaluation.

---

## Dataset

The data for this project was sourced from the "Credit Score Classification" dataset on Kaggle. You can find it here: [https://www.kaggle.com/datasets/parisrohan/credit-score-classification/data](https://www.kaggle.com/datasets/parisrohan/credit-score-classification/data)

---

## Methodology & Workflow

The project followed a structured data science workflow:

1.  **Data Cleaning & Preprocessing:**
    * Removed unnecessary identifier columns (`ID`, `Customer_ID`, `Name`, `SSN`).
    * Cleaned "dirty" numerical columns that were incorrectly typed as objects (e.g., `Annual_Income`, `Outstanding_Debt`).
    * Handled invalid data points, such as negative ages, by replacing them with `NaN`.
    * Performed median imputation for all missing numerical values.
    * Engineered a new feature, `Credit_History_Age_Months`, from a complex string column.

2.  **Feature Engineering:**
    * Utilized `CountVectorizer` to handle the multi-value `Type_of_Loan` column, creating binary features for each loan type.
    * Applied **Label Encoding** to the target variable (`Credit_Score`) and **One-Hot Encoding** to all remaining categorical features (`Occupation`, `Credit_Mix`, etc.).

3.  **Modeling & Evaluation:**
    * Created a validation set using `train_test_split` (80/20 split) to reliably evaluate model performance.
    * Applied `StandardScaler` to the features to standardize their scale.
    * Trained a `RandomForestClassifier` model on the prepared data.
    * Evaluated the model using Accuracy, Precision, Recall, and F1-Score.

---

## Model Performance

The model was evaluated on a 20% validation set held out from the original training data. The performance was as follows:

* **Overall Accuracy:** **78.83%**

**Classification Report on Validation Set:**

| Class      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Poor (0)   | **0.79** | **0.80** | **0.79** |
| Standard(1)| **0.81** | **0.81**| **0.81**|
| Good (2)   | **0.72** | **0.71** | **0.72** |


---

## Reflections & Future Work

This project was a comprehensive exercise in handling a realistic, messy dataset. A key challenge was cleaning the numerous data types and engineering features from complex strings like `Type_of_Loan` and `Credit_History_Age`.

**Future improvements could include:**
* Experimenting with other models like Gradient Boosting (XGBoost, LightGBM) to potentially improve the F1-score for the 'Good' credit class.
* Performing more in-depth feature selection to see if a simpler model could achieve similar performance.

---

## Acknowledgements

This project was completed as a guided project under the mentorship of 'Professor Axiom' (a Google AI) to solidify foundational machine learning concepts.
