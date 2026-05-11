import numpy as np
import sys
from DecisionTreeNumPy import DecisionTree
from tqdm import tqdm


class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, max_features='sqrt', bootstrap=True, random_state=None):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.trees = []
        self.features_ = []

    def _max_features_count(self, n_features):
        mf = self.max_features
        if isinstance(mf, int):
            return max(1, min(mf, n_features))
        if isinstance(mf, float):
            return max(1, int(mf * n_features))
        if isinstance(mf, str):
            if mf == 'sqrt':
                return max(1, int(np.sqrt(n_features)))
            if mf == 'log2':
                return max(1, int(np.log2(n_features)))
        return n_features

    def fit(self, X, y, verbose=False):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        m = self._max_features_count(n_features)
        self.trees = []
        self.features_ = []
        iterable = range(self.n_estimators)
        if verbose:
            iterable = tqdm(iterable, desc="Training Forest", unit="tree")
        for i in iterable:
            if self.bootstrap:
                idxs = self.rng.randint(0, n_samples, n_samples)
            else:
                idxs = np.arange(n_samples)
            feat_idxs = self.rng.choice(n_features, m, replace=False)

            X_boot = X[idxs][:, feat_idxs]
            y_boot = y[idxs]

            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            self.features_.append(feat_idxs)

    def predict(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        preds = np.empty((self.n_estimators, n_samples), dtype=int)
        for i, tree in enumerate(self.trees):
            feat = self.features_[i]
            X_sub = X[:, feat]
            preds[i] = tree.predict(X_sub)

        # majority vote
        y_pred = np.empty(n_samples, dtype=int)
        for j in range(n_samples):
            vals, counts = np.unique(preds[:, j], return_counts=True)
            y_pred[j] = vals[np.argmax(counts)]
        return y_pred

