import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, recall_score, f1_score, accuracy_score, precision_score


def load_and_preprocess(path="insurance.csv"):
    df = pd.read_csv(path)
    df = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)
    return df


def regression_pipeline(df):
    X = df.drop(columns=["charges"])
    y = df["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Linear Regression results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R^2: {r2:.4f}")
    return lr, X_test, y_test, y_pred


def classification_pipeline(df, X_test, y_test, y_pred_regression):
    # Create a binary target using the median of charges
    threshold = df["charges"].median()
    y_binary = (df["charges"] > threshold).astype(int)
    X = df.drop(columns=["charges"])

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_c, y_train_c)
    y_pred_clf = clf.predict(X_test_c)

    print("\nLogistic Regression (binary target) results:")
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test_c, y_pred_clf))
    print(f"  Recall: {recall_score(y_test_c, y_pred_clf):.4f}")
    print(f"  F1 Score: {f1_score(y_test_c, y_pred_clf):.4f}")
    print(f"  Accuracy: {accuracy_score(y_test_c, y_pred_clf):.4f}")
    print(f"  Precision: {precision_score(y_test_c, y_pred_clf):.4f}")

    # Also compute metrics by thresholding regression predictions
    y_test_thresholded = (y_test > threshold).astype(int)
    y_pred_thresholded = (y_pred_regression > threshold).astype(int)

    print("\nClassification metrics from Linear Regression predictions (thresholded at median):")
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test_thresholded, y_pred_thresholded))
    print(f"  Recall: {recall_score(y_test_thresholded, y_pred_thresholded):.4f}")
    print(f"  F1 Score: {f1_score(y_test_thresholded, y_pred_thresholded):.4f}")
    print(f"  Accuracy: {accuracy_score(y_test_thresholded, y_pred_thresholded):.4f}")
    print(f"  Precision: {precision_score(y_test_thresholded, y_pred_thresholded):.4f}")


def main():
    df = load_and_preprocess("insurance.csv")
    lr_model, X_test, y_test, y_pred = regression_pipeline(df)
    classification_pipeline(df, X_test, y_test, y_pred)


if __name__ == "__main__":
    main()
