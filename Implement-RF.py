import numpy as np
import os
from RandomForestNumpy import RandomForest


def load_wine_quality(folder):
    files = ["winequality-red.csv", "winequality-white.csv"]
    parts = []
    for fn in files:
        path = os.path.join(folder, fn)
        data = np.genfromtxt(path, delimiter=';', skip_header=1)
        parts.append(data)
    return np.vstack(parts)


def f1_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, 'wine+quality')
    data = load_wine_quality(data_dir)
    X = data[:, :-1]
    y_raw = data[:, -1].astype(int)

    # Binary label: quality >= 7 => good (1), else 0
    y = (y_raw >= 7).astype(int)

    rng = np.random.RandomState(123)
    idx = rng.permutation(len(X))
    X = X[idx]
    y = y[idx]

    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    rf = RandomForest(n_estimators=100, max_depth=12, min_samples_split=5, max_features='sqrt', random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    precision, recall, f1 = f1_score(y_test, y_pred)

    print(f"Samples: total={len(X)}, train={len(X_train)}, test={len(X_test)}")
    print(f"Positive class (good wine) in test: {np.sum(y_test==1)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")


if __name__ == '__main__':
    main()
