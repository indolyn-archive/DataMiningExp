from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def find_dataset() -> Path:
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "实验二 逻辑回归 Iris数据集" / "Iris数据集" / "iris.csv",
        base_dir / "实验二 逻辑回归 Iris数据集" / "Iris数据集" / "Iris.data",
        base_dir / "实验二 逻辑回归 Iris数据集" / "Iris数据集" / "iris.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("找不到 iris 数据文件，请检查 exp\\02\\实验二 逻辑回归 Iris数据集\\Iris数据集")


def load_iris_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "Species" not in df.columns:
            raise ValueError("CSV 文件缺少 Species 列")
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        return df

    # 兼容 .data / .txt 格式
    df = pd.read_csv(
        path,
        header=None,
        names=["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species"],
    )
    return df


def main() -> None:
    dataset_path = find_dataset()
    print(f"Using dataset: {dataset_path}")

    df = load_iris_data(dataset_path)
    print("\nData preview:")
    print(df.head())
    print("\nData info:")
    print(df.info())
    print("\nClass distribution:")
    print(df["Species"].value_counts())

    # 标签编码
    le = LabelEncoder()
    df["Species"] = le.fit_transform(df["Species"].astype(str))

    X = df.drop(columns=["Species"])
    y = df["Species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 逻辑回归需要较稳定的数值尺度，这里用标准化 + 逻辑回归组成流水线
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)

    print("\nAccuracy:", acc)
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(report)

    out_dir = Path(__file__).resolve().parent

    # 特征相关性热力图
    plt.figure(figsize=(7, 6))
    corr = X.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    corr_path = out_dir / "feature_correlation_heatmap.png"
    plt.savefig(corr_path, dpi=200)
    plt.close()
    print(f"Saved feature correlation heatmap to: {corr_path}")

    # 各特征箱线图
    long_df = X.copy()
    long_df["Species"] = le.inverse_transform(y)
    plot_df = long_df.melt(id_vars="Species", var_name="Feature", value_name="Value")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_df, x="Feature", y="Value", hue="Species")
    plt.title("Feature Boxplots by Class")
    plt.tight_layout()
    box_path = out_dir / "feature_boxplots.png"
    plt.savefig(box_path, dpi=200)
    plt.close()
    print(f"Saved feature boxplots to: {box_path}")

    # 混淆矩阵图
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"\nSaved confusion matrix to: {cm_path}")

    # PCA 二维展示
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", s=35)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Iris PCA Visualization")
    plt.legend(*scatter.legend_elements(), title="Class", loc="best")
    plt.tight_layout()
    pca_path = out_dir / "pca_visualization.png"
    plt.savefig(pca_path, dpi=200)
    plt.close()
    print(f"Saved PCA visualization to: {pca_path}")

    # 多分类 ROC 曲线
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    y_score = model.predict_proba(X_test)
    plt.figure(figsize=(7, 6))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for i, color in enumerate(colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{le.classes_[i]} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = out_dir / "roc_curves.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"Saved ROC curves to: {roc_path}")


if __name__ == "__main__":
    main()
