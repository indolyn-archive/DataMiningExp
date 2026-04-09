from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "实验五 数据集" / "Iris数据集"
CSV_PATH = DATA_DIR / "iris.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    le = LabelEncoder()
    encoded = df.copy()
    encoded["Species"] = le.fit_transform(encoded["Species"].astype(str))
    return encoded, le


def save_confusion_matrix(cm, labels, path: Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_pca_plot(X, y, labels, path: Path) -> None:
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", s=35)
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Iris PCA Visualization")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def evaluate_model(name, model, X_train, X_test, y_train, y_test, labels, out_dir: Path):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=labels)

    print(f"\n{name} Accuracy: {acc:.4f}")
    print(report)

    cm_path = out_dir / f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
    save_confusion_matrix(cm, labels, cm_path, f"{name} Confusion Matrix")
    print("Saved:", cm_path)

    return acc, cm, report


def main() -> None:
    out_dir = BASE_DIR
    df = load_data()

    print("Data preview:")
    print(df.head())
    print("\nClass distribution:")
    print(df["Species"].value_counts())

    encoded, le = encode_labels(df)
    X = encoded.drop(columns=["Species"])
    y = encoded["Species"]
    labels = list(le.classes_)

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Species", order=df["Species"].value_counts().index)
    plt.title("Class Distribution")
    plt.tight_layout()
    dist_path = out_dir / "iris_class_distribution.png"
    plt.savefig(dist_path, dpi=200)
    plt.close()
    print("Saved:", dist_path)

    save_pca_plot(X, y, labels, out_dir / "iris_pca.png")
    print("Saved:", out_dir / "iris_pca.png")

    # GaussianNB 直接适合连续特征
    X_train_g, X_test_g, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    gnb = GaussianNB()
    g_acc, _, _ = evaluate_model("GaussianNB", gnb, X_train_g, X_test_g, y_train, y_test, labels, out_dir)

    # MultinomialNB 和 BernoulliNB 更适合离散/二值特征，所以这里做额外变换后再比较
    mm = MinMaxScaler()
    X_mm = pd.DataFrame(mm.fit_transform(X), columns=X.columns)
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_mm, y, test_size=0.2, random_state=42, stratify=y
    )
    mnb = MultinomialNB()
    m_acc, _, _ = evaluate_model("MultinomialNB", mnb, X_train_m, X_test_m, y_train_m, y_test_m, labels, out_dir)

    # BernoulliNB 使用二值化数据
    X_bin = (X_mm > 0.5).astype(int)
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_bin, y, test_size=0.2, random_state=42, stratify=y
    )
    bnb = BernoulliNB()
    b_acc, _, _ = evaluate_model("BernoulliNB", bnb, X_train_b, X_test_b, y_train_b, y_test_b, labels, out_dir)

    # 准确率对比图
    plt.figure(figsize=(7, 5))
    methods = ["GaussianNB", "MultinomialNB", "BernoulliNB"]
    accuracies = [g_acc, m_acc, b_acc]
    sns.barplot(x=methods, y=accuracies, palette="Set2")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Naive Bayes Model Comparison")
    plt.tight_layout()
    cmp_path = out_dir / "naive_bayes_compare.png"
    plt.savefig(cmp_path, dpi=200)
    plt.close()
    print("Saved:", cmp_path)


if __name__ == "__main__":
    main()
