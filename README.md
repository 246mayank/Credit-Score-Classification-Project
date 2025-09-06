# Credit Score Classification Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-Fast%20Boosting-green?style=for-the-badge)
![Kaggle](https://img.shields.io/badge/Kaggle-API%20Integrated-20BEFF?style=for-the-badge&logo=kaggle)

**🚀 A comprehensive machine learning project delivering robust credit score classification with state-of-the-art models**

[🔍 **Explore the Code**](./credit_score_classification.ipynb) • [📊 **View Dataset**](https://www.kaggle.com/datasets/mavimayank/train-and-test-creditscore) • [📈 **See Results**](#-model-performance)

</div>

---

## 🌟 Project Overview

This project delivers a robust system for predicting credit scores by classifying customers into **Poor**, **Standard**, and **Good** risk levels. It uses a multi-model approach, including Random Forest, XGBoost, and LightGBM, and features a fully automated data pipeline that integrates with the Kaggle API.

The project is available in two formats:
-   **Jupyter Notebook**: For learning, exploration, and detailed analysis.
-   **Python Script**: For production, automation, and fast execution.

For a detailed technical breakdown, see the [**Project Report**](./project_report.txt).

---

## ✨ Features

-   **Multi-Model Training**: Compares Random Forest, XGBoost, and LightGBM.
-   **Hyperparameter Tuning**: Uses GridSearchCV for optimal model performance.
-   **Automated Data Pipeline**: Downloads and preprocesses data automatically.
-   **Dual Implementation**: Notebook for analysis, script for production.

---

## � Quick Start

### 1. Prerequisites
```bash
# Python 3.8+ is required
pip install -r requirements.txt
```

### 2. Kaggle API Setup
1.  Create a Kaggle account and download your `kaggle.json` API token.
2.  Place it in `C:\Users\{username}\.kaggle\` (Windows) or `~/.kaggle/` (Mac/Linux).

### 3. Run the Project

#### **For Learning & Exploration (Recommended):**
```bash
# Launch Jupyter and open the notebook
jupyter notebook credit_score_classification.ipynb
```

#### **For Automation & Production:**
```bash
# Run the complete pipeline from your terminal
python Credit_Score_classsification.py
```

---

## 📊 Model Performance

Based on the latest run, the model accuracies are as follows. Random Forest was the top performer.

| Model         | Test Accuracy |
|---------------|---------------|
| **Random Forest** | **~78.8%**    |
| LightGBM      | ~78.4%        |
| XGBoost       | ~75.9%        |

---



</div>

</div>

**Key Advantages of Jupyter notebook:**
- Comprehensive Documentation: Markdown cells explain every step
- Rich Visualizations: Plots, charts, and data exploration graphics  
- Interactive Analysis: Run cells individually, modify parameters easily
- Educational Design: Perfect for understanding ML concepts
- Advanced Hyperparameter Tuning: More extensive parameter grids
- Smart Environment Detection: Works seamlessly in Kaggle or local setup


**Key Advantages of python script:**
- Fast Execution: Single command, complete pipeline
- Automation-Ready: Perfect for scheduled jobs and batch processing
- Lightweight: Minimal dependencies, clean execution
- Production-Friendly: Easy integration into larger systems
- Streamlined: No UI overhead, pure computation

### 💡 **Our Recommendation:**

<div align="center">

| Use Case | Best Choice | Reason |
|:---------------:|:------------------:|:-------------:|
| Learning ML | Jupyter Notebook | Rich documentation & visualizations |
| Building Portfolio | Jupyter Notebook | Professional presentation |
| Production Deployment | Python Script | Automation & performance |
| Research & Analysis | Jupyter Notebook | Interactive exploration |
| Quick Predictions | Python Script | Fast execution |

</div>

> **Pro Tip**: Start with the **Jupyter Notebook** to understand the methodology, then use the **Python Script** for production deployment!

---
##  Data Pipeline

<div align="center">

```mermaid
graph LR
    A[Kaggle API] --> B[Raw Data]
    B --> C[Data Cleaning]
    C --> D[Feature Engineering]
    D --> E[Scaling]
    E --> F[Model Training]
    F --> G[Evaluation]
    G --> H[Predictions]
```

</div>

### 🛠️ **Preprocessing Steps**

1. **Identifier Removal**: Drop non-predictive columns (ID, Name, SSN)
2. **Numeric Cleaning**: Convert text-based numbers to proper format
3. **Data Validation**: Correct invalid entries (age outliers, etc.)
4. **Missing Value Imputation**: Smart median-based filling
5. **Feature Conversion**: Transform credit history to numerical months
6. **Loan Type Encoding**: Multi-hot encoding for multiple loan types
7. **Categorical Encoding**: One-hot encoding for remaining features
8. **Feature Scaling**: StandardScaler normalization

---

## 🛠️ Technical Implementation

### **Model Evaluation Methodology**
- Stratified Splits: Maintaining class distribution in train/validation
- Cross-Validation: 3-fold CV during hyperparameter tuning
- Multiple Metrics: Accuracy, Precision, Recall, F1-Score
- Confusion Matrices: Detailed class-wise performance analysis
- Automated Selection: Best model chosen by validation accuracy

### **Hyperparameter Optimization**
- Random Forest: n_estimators, max_depth, min_samples_split
- XGBoost: learning_rate, max_depth, n_estimators
- LightGBM: learning_rate, num_leaves, n_estimators

---

## 📚 Documentation & Resources

### **Project Documentation**
- [**Dataset Information**](https://www.kaggle.com/datasets/mavimayank/train-and-test-creditscore)
- [**Model Comparison Guide**](./docs/model_comparison.md)
- [**Technical Architecture**](./docs/architecture.md)

### **Learning Resources**
- [**Machine Learning Best Practices**](https://scikit-learn.org/stable/tutorial/index.html)
- [**XGBoost Documentation**](https://xgboost.readthedocs.io/)
- [**LightGBM Guide**](https://lightgbm.readthedocs.io/)

---



### **Bug Reports**
Found a bug? Please open an [issue](../../issues) with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details

### **Feature Requests**
Have an idea? We'd love to hear it! Open an [issue](../../issues) describing:
- The feature you'd like to see
- Why it would be useful
- Potential implementation approach

---

<div align="center">

### 📬 **Contact & Connect**

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=for-the-badge&logo=github)](https://github.com/246mayank)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/246mayank)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/mayank-singh-789719255/)

**⭐ Star this repository if you found it helpful!**

</div>


