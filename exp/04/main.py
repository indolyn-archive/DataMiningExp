from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.tree import DecisionTreeClassifier, plot_tree


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "实验四 决策树和随机森林 cardata数据集" / "car_data数据集"
CSV_PATH = DATA_DIR / "car_data数据集.CSV"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    if "Column1" in df.columns:
        df = df.drop(index=0).reset_index(drop=True)
    if df.iloc[0].tolist() == ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]:
        df = df.drop(index=0).reset_index(drop=True)
    df.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    return df


def encode_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    encoded = df.copy()
    class_encoder = LabelEncoder()
    for col in encoded.columns:
        encoded[col] = LabelEncoder().fit_transform(encoded[col].astype(str))
    encoded["class"] = class_encoder.fit_transform(df["class"].astype(str))
    return encoded, class_encoder


def save_confusion_matrix(cm: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_roc_curve(y_true: np.ndarray, y_score: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    y_true_bin = label_binarize(y_true, classes=list(range(len(labels))))
    plt.figure(figsize=(8, 6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"{label} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    out_dir = BASE_DIR
    df = load_data()
    print("Data preview:")
    print(df.head())
    print("\nClass distribution:")
    print(df["class"].value_counts())

    encoded, class_le = encode_frame(df)
    X = encoded.drop(columns=["class"])
    y = encoded["class"]
    class_names = list(class_le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="class", order=df["class"].value_counts().index)
    plt.title("Class Distribution")
    plt.tight_layout()
    dist_path = out_dir / "car_class_distribution.png"
    plt.savefig(dist_path, dpi=200)
    plt.close()
    print("Saved:", dist_path)

    tree_model = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=6)
    tree_model.fit(X_train, y_train)
    tree_pred = tree_model.predict(X_test)
    tree_prob = tree_model.predict_proba(X_test)
    tree_acc = accuracy_score(y_test, tree_pred)
    tree_cm = confusion_matrix(y_test, tree_pred)
    tree_report = classification_report(y_test, tree_pred, target_names=class_names)
    print("\nDecision Tree Accuracy:", tree_acc)
    print(tree_report)

    save_confusion_matrix(tree_cm, class_names, out_dir / "tree_confusion_matrix.png", "Decision Tree Confusion Matrix")
    save_roc_curve(y_test.to_numpy(), tree_prob, class_names, out_dir / "tree_roc_curve.png", "Decision Tree ROC Curve")

    plt.figure(figsize=(16, 10))
    plot_tree(
        tree_model,
        feature_names=X.columns,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title("Decision Tree Structure")
    plt.tight_layout()
    plt.savefig(out_dir / "decision_tree.png", dpi=200)
    plt.close()

    rf_model = RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        random_state=42,
        max_depth=None,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_cm = confusion_matrix(y_test, rf_pred)
    rf_report = classification_report(y_test, rf_pred, target_names=class_names)
    print("\nRandom Forest Accuracy:", rf_acc)
    print(rf_report)

    save_confusion_matrix(rf_cm, class_names, out_dir / "rf_confusion_matrix.png", "Random Forest Confusion Matrix")
    save_roc_curve(y_test.to_numpy(), rf_prob, class_names, out_dir / "rf_roc_curve.png", "Random Forest ROC Curve")

    importances = rf_model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[order], y=X.columns[order], orient="h")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(out_dir / "rf_feature_importance.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(x=["Decision Tree", "Random Forest"], y=[tree_acc, rf_acc], palette="Set2")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(out_dir / "model_accuracy_compare.png", dpi=200)
    plt.close()

    print("Saved all figures in:", out_dir)


if __name__ == "__main__":
    main()
