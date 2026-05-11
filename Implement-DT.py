import numpy as np
from DecisionTreeNumPy import DecisionTree
import os


def load_wine_quality(folder):
    files = [("winequality-red.csv", 0), ("winequality-white.csv", 1)] # 0: đỏ, 1: trắng
    parts = []
    for fn, color_val in files:
        path = os.path.join(folder, fn)
        data = np.genfromtxt(path, delimiter=';', skip_header=1)
        # Tạo thêm 1 cột chứa giá trị color_val
        color_col = np.full((data.shape[0], 1), color_val)
        data_with_color = np.hstack((data[:, :-1], color_col, data[:, -1:]))
        parts.append(data_with_color)
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
    data = load_wine_quality(data_dir) # Sử dụng hàm load đã thêm cột color
    
    X = data[:, :-1]
    y_raw = data[:, -1].astype(int)
    y = (y_raw >= 7).astype(int)

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # --- BƯỚC QUAN TRỌNG: OVERSAMPLING ---
    X_1 = X_train[y_train == 1]
    y_1 = y_train[y_train == 1]
    # Nhân đôi lớp 1 để tăng trọng số cho Recall
    X_train_resampled = np.vstack([X_train, X_1])
    y_train_resampled = np.concatenate([y_train, y_1])

    # Điều chỉnh tham số: sâu hơn, nhạy hơn
    clf = DecisionTree(max_depth=15, min_samples_split=2)
    clf.fit(X_train_resampled, y_train_resampled)
    
    y_pred = clf.predict(X_test)
    precision, recall, f1 = f1_score(y_test, y_pred)

    print(f"--- Kết quả sau điều chỉnh ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:   {recall:.4f}")
    print(f"F1 score: {f1:.4f}")


if __name__ == '__main__':
    main()
