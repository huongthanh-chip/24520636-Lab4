import numpy as np
import os
from RandomForestNumpy import RandomForest
from time import time


def load_wine_quality(folder):
    files = ["winequality-red.csv", "winequality-white.csv"]
    parts = []
    for fn in files:
        path = os.path.join(folder, fn)
        data = np.genfromtxt(path, delimiter=';', skip_header=1)
        parts.append(data)
    return np.vstack(parts)


def metrics(y_true, y_pred):
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
    y = (y_raw >= 7).astype(int)

    rng = np.random.RandomState(123)
    idx = rng.permutation(len(X))
    X = X[idx]
    y = y[idx]

    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    param_grid = {
        'n_estimators': [10, 50],
        'max_depth': [8, 12, 16, None],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2', None]
    }

    results = []
    total = (len(param_grid['n_estimators']) * len(param_grid['max_depth']) *
             len(param_grid['min_samples_split']) * len(param_grid['max_features']))
    cnt = 0
    start_all = time()
    for n in param_grid['n_estimators']:
        for md in param_grid['max_depth']:
            for mss in param_grid['min_samples_split']:
                for mf in param_grid['max_features']:
                    cnt += 1
                    t0 = time()
                    rf = RandomForest(n_estimators=n, max_depth=md, min_samples_split=mss,
                                      max_features=mf, random_state=42)
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_test)
                    precision, recall, f1 = metrics(y_test, y_pred)
                    t1 = time()
                    results.append({'n_estimators': n, 'max_depth': md, 'min_samples_split': mss,
                                    'max_features': mf, 'precision': precision, 'recall': recall,
                                    'f1': f1, 'time': t1 - t0})
                    print(f"[{cnt}/{total}] n={n} md={md} mss={mss} mf={mf} -> P={precision:.4f} R={recall:.4f} F1={f1:.4f} time={t1-t0:.2f}s")

    elapsed = time() - start_all
    print(f"Grid search done in {elapsed:.1f}s, tested {len(results)} configs")

    # best by F1
    best_f1 = sorted(results, key=lambda r: r['f1'], reverse=True)[:5]
    print('\nTop 5 by F1:')
    for r in best_f1:
        print(r)

    # best by recall
    best_recall = sorted(results, key=lambda r: r['recall'], reverse=True)[:5]
    print('\nTop 5 by Recall:')
    for r in best_recall:
        print(r)


if __name__ == '__main__':
    main()
