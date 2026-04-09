from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


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
    raise FileNotFoundError("找不到 iris 数据文件")


def load_iris_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        return df
    return pd.read_csv(
        path,
        header=None,
        names=["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species"],
    )


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class SoftmaxRegression:
    def __init__(self, lr: float = 0.1, epochs: int = 2000, reg: float = 0.0):
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.W = None
        self.b = None
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftmaxRegression":
        n_samples, n_features = X.shape
        n_classes = int(np.max(y)) + 1
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros((1, n_classes))
        y_onehot = np.eye(n_classes)[y]

        for _ in range(self.epochs):
            scores = X @ self.W + self.b
            probs = softmax(scores)
            loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1))
            loss += 0.5 * self.reg * np.sum(self.W * self.W)
            self.loss_history.append(loss)

            grad_scores = (probs - y_onehot) / n_samples
            grad_W = X.T @ grad_scores + self.reg * self.W
            grad_b = np.sum(grad_scores, axis=0, keepdims=True)

            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(X @ self.W + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    dataset_path = find_dataset()
    df = load_iris_data(dataset_path)

    le = LabelEncoder()
    df["Species"] = le.fit_transform(df["Species"].astype(str))

    X = df.drop(columns=["Species"]).values
    y = df["Species"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SoftmaxRegression(lr=0.15, epochs=1500, reg=0.001)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)

    print("Improved softmax regression")
    print("Accuracy:", acc)
    print("Confusion matrix:\n", cm)
    print(report)

    # 损失曲线
    plt.figure(figsize=(7, 5))
    plt.plot(model.loss_history, color="tab:red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.tight_layout()
    loss_path = out_dir / "improved_loss_curve.png"
    plt.savefig(loss_path, dpi=200)
    plt.close()

    # 预测结果对比图
    plt.figure(figsize=(7, 6))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    markers = ["o", "s", "^"]
    for cls in np.unique(y_test):
        idx = y_test == cls
        plt.scatter(
            X_test[idx, 2],
            X_test[idx, 3],
            c=colors[cls],
            marker=markers[cls],
            label=f"True {le.classes_[cls]}",
            edgecolors="black",
            s=70,
        )
    wrong = y_pred != y_test
    if np.any(wrong):
        plt.scatter(
            X_test[wrong, 2],
            X_test[wrong, 3],
            facecolors="none",
            edgecolors="red",
            s=140,
            linewidths=2,
            label="Misclassified",
        )
    plt.xlabel("Petal.Length (standardized)")
    plt.ylabel("Petal.Width (standardized)")
    plt.title("Improved Model Prediction View")
    plt.legend()
    plt.tight_layout()
    pred_path = out_dir / "improved_prediction_view.png"
    plt.savefig(pred_path, dpi=200)
    plt.close()

    # 混淆矩阵图
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(le.classes_)), le.classes_)
    plt.yticks(range(len(le.classes_)), le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Improved Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    cm_path = out_dir / "improved_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    print("Saved:", loss_path)
    print("Saved:", pred_path)
    print("Saved:", cm_path)


if __name__ == "__main__":
    main()
