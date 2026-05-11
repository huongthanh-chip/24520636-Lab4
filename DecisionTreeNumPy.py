import numpy as np
import sys

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth if max_depth is None or max_depth > 0 else None
        self.min_samples_split = max(2, min_samples_split)
        self.root = None

    def fit(self, X, y, verbose=False):
        X = np.asarray(X)
        y = np.asarray(y)
        self.verbose = verbose
        self.node_count = 0
        if self.verbose:
            print("Training Decision Tree...")
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self.root = self._grow_tree(X, y)
        if self.verbose:
            sys.stdout.write(f"\rDone! Total nodes created: {self.node_count}\n")

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x, self.root) for x in X])

    def _gini(self, y):
        m = y.size
        if m == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        prob = counts / m
        return 1.0 - np.sum(prob ** 2)

    def _most_common_label(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def _best_split(self, X, y):
        m, n = X.shape
        if m < self.min_samples_split:
            return None, None

        best_gini = 1.0
        best_idx, best_thr = None, None

        parent_gini = self._gini(y)
        if parent_gini == 0:
            return None, None

        for feature_idx in range(n):
            thresholds = np.unique(X[:, feature_idx])
            for thr in thresholds:
                left_mask = X[:, feature_idx] <= thr
                right_mask = ~left_mask
                y_left = y[left_mask]
                y_right = y[right_mask]
                if y_left.size == 0 or y_right.size == 0:
                    continue
                gini_left = self._gini(y_left)
                gini_right = self._gini(y_right)
                weighted_gini = (y_left.size * gini_left + y_right.size * gini_right) / m
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_idx = feature_idx
                    best_thr = thr

        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        if self.verbose:
            self.node_count += 1
            spinner = ['|', '/', '-', '\\']
            sys.stdout.write(f"\r{spinner[self.node_count % 4]} Nodes grown: {self.node_count}")
            sys.stdout.flush()

        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        node = {}

        if (self.max_depth is not None and depth >= self.max_depth) or num_samples < self.min_samples_split or num_labels == 1:
            node['type'] = 'leaf'
            node['value'] = self._most_common_label(y)
            return node

        feat_idx, thr = self._best_split(X, y)
        if feat_idx is None:
            node['type'] = 'leaf'
            node['value'] = self._most_common_label(y)
            return node

        left_mask = X[:, feat_idx] <= thr
        right_mask = ~left_mask

        node['type'] = 'node'
        node['feature'] = feat_idx
        node['threshold'] = thr
        node['left'] = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        node['right'] = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _predict_one(self, x, node):
        if node['type'] == 'leaf':
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])

