import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support


def load_wine_quality(folder):
    files = ["winequality-red.csv", "winequality-white.csv"]
    parts = []
    for fn in files:
        path = os.path.join(folder, fn)
        data = np.genfromtxt(path, delimiter=';', skip_header=1)
        parts.append(data)
    return np.vstack(parts)


def report(name, y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    print(f"{name}: Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, 'wine+quality')
    data = load_wine_quality(data_dir)
    X = data[:, :-1]
    y_raw = data[:, -1].astype(int)
    y = (y_raw >= 7).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_dt = dt.predict(X_test)
    report('DecisionTree (sklearn)', y_test, y_dt)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_rf = rf.predict(X_test)
    report('RandomForest (sklearn)', y_test, y_rf)


if __name__ == '__main__':
    main()
