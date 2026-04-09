from __future__ import annotations

import gzip
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
MNIST_DIR = BASE_DIR / "实验三 支持向量机 mnist数据集" / "MNIST1"


def load_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image magic number: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num_images, rows * cols)


def load_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label magic number: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = load_idx_images(MNIST_DIR / "train-images-idx3-ubyte.gz")
    y_train = load_idx_labels(MNIST_DIR / "train-labels-idx1-ubyte.gz")
    x_test = load_idx_images(MNIST_DIR / "t10k-images-idx3-ubyte.gz")
    y_test = load_idx_labels(MNIST_DIR / "t10k-labels-idx1-ubyte.gz")
    return x_train, y_train, x_test, y_test


def save_sample_grid(x: np.ndarray, y: np.ndarray, path: Path, title: str, n: int = 16) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle(title)
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"Label: {y[i]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def save_confusion_matrix(cm: np.ndarray, labels: list[int], path: Path, title: str) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_pca_plot(x: np.ndarray, y: np.ndarray, path: Path, title: str, sample_size: int = 2000) -> None:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x), size=min(sample_size, len(x)), replace=False)
    x_sub = x[idx]
    y_sub = y[idx]
    x_scaled = StandardScaler().fit_transform(x_sub)
    x_pca = PCA(n_components=2, random_state=42).fit_transform(x_scaled)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y_sub, cmap="tab10", s=12, alpha=0.8)
    plt.legend(*scatter.legend_elements(), title="Digit", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    out_dir = BASE_DIR
    x_train, y_train, x_test, y_test = load_mnist()

    print("Train shape:", x_train.shape, y_train.shape)
    print("Test shape:", x_test.shape, y_test.shape)
    print("Train label distribution:", np.bincount(y_train))

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # 为了让 SVM 在作业环境中能较快跑完，这里先抽取一部分训练数据做调参
    x_sub, _, y_sub, _ = train_test_split(
        x_train, y_train, train_size=3000, random_state=42, stratify=y_train
    )

    sample_grid_path = out_dir / "mnist_samples.png"
    save_sample_grid(x_train[:16], y_train[:16], sample_grid_path, "MNIST Sample Grid")
    print("Saved:", sample_grid_path)

    pca_path = out_dir / "mnist_pca.png"
    save_pca_plot(x_sub, y_sub, pca_path, "MNIST PCA Visualization")
    print("Saved:", pca_path)

    # 基础模型
    base_model = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("svc", SVC(kernel="rbf", C=5.0, gamma="scale")),
        ]
    )
    base_model.fit(x_sub, y_sub)
    base_pred = base_model.predict(x_test)
    base_acc = accuracy_score(y_test, base_pred)
    base_cm = confusion_matrix(y_test, base_pred)
    base_report = classification_report(y_test, base_pred, digits=4)

    print("\nBase SVM Accuracy:", base_acc)
    print(base_report)

    base_cm_path = out_dir / "svm_confusion_matrix.png"
    save_confusion_matrix(base_cm, list(range(10)), base_cm_path, "Base SVM Confusion Matrix")
    print("Saved:", base_cm_path)

    # 参数字典 + 交叉验证
    param_grid = {
        "svc__kernel": ["rbf", "linear"],
        "svc__C": [1, 5, 10],
        "svc__gamma": ["scale", 0.01],
    }

    grid_model = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("svc", SVC()),
        ]
    )

    grid = GridSearchCV(
        grid_model,
        param_grid=param_grid,
        cv=2,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(x_sub, y_sub)

    print("\nBest params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    best_model = grid.best_estimator_
    best_pred = best_model.predict(x_test)
    best_acc = accuracy_score(y_test, best_pred)
    best_cm = confusion_matrix(y_test, best_pred)
    best_report = classification_report(y_test, best_pred, digits=4)

    print("\nBest SVM Accuracy:", best_acc)
    print(best_report)

    best_cm_path = out_dir / "svm_best_confusion_matrix.png"
    save_confusion_matrix(best_cm, list(range(10)), best_cm_path, "Best SVM Confusion Matrix")
    print("Saved:", best_cm_path)

    # 参数对比图
    results = grid.cv_results_
    mean_scores = results["mean_test_score"]
    labels = [
        f'{p["svc__kernel"]}\nC={p["svc__C"]}\ng={p["svc__gamma"]}'
        for p in results["params"]
    ]
    plt.figure(figsize=(12, 6))
    order = np.argsort(mean_scores)[::-1][:12]
    plt.bar(range(len(order)), mean_scores[order], color="steelblue")
    plt.xticks(range(len(order)), [labels[i] for i in order], rotation=45, ha="right")
    plt.ylabel("Mean CV Score")
    plt.title("Grid Search Top Parameters")
    plt.tight_layout()
    grid_path = out_dir / "svm_grid_search_top.png"
    plt.savefig(grid_path, dpi=200)
    plt.close()
    print("Saved:", grid_path)


if __name__ == "__main__":
    main()
