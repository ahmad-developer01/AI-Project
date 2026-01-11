
# insurance-charges-analysis

A small analysis pipeline to predict and evaluate insurance `charges` using regression and classification models.

This repository contains a small analysis pipeline for the common `insurance.csv` dataset (expected in the repository root).

Overview
- `analysis_insurance.py`: Loads the dataset, preprocesses categorical variables, trains a Linear Regression model to predict insurance `charges`, and trains a Logistic Regression classifier on a binary target (charges > median). The script prints regression and classification metrics.
- `requirements.txt`: Minimal dependencies.

What the script prints
- Linear Regression: Mean Squared Error (MSE) and R² score.
- Logistic Regression (binary target): Confusion Matrix, Recall, F1 score, Accuracy, Precision.
- Additional classification metrics obtained by thresholding the regression predictions at the median charge.

Dataset expectation
- The script expects `insurance.csv` in the repository root. Typical columns: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`.

Quick start
1. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

2. Run the analysis:

```powershell
python analysis_insurance.py
```

Notes & extensions
- You can change the binary threshold from median to any other value inside `analysis_insurance.py`.
- Replace or extend models (e.g., RandomForestRegressor, XGBoost) for improved performance.
- Consider saving trained models or outputting results to CSV for further analysis.

License
- Feel free to use and adapt this code for educational or experimental purposes.

