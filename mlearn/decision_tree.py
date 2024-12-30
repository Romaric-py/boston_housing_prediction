import numpy as np
from collections import Counter


class Node:
    """
    Represents a single node in a decision tree, which can be either a decision 
    node with a feature and threshold or a leaf node with a value.
    """
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None




class DecisionTreeClassifier:
    """
    A classifier that uses a decision tree to make predictions based on input features.
    """
    
    def __init__(self, n_features=None, max_depth=10, min_samples_split=3, random_state=None):
        self.n_features = n_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.random_state = random_state
        self._is_fitted = False
    
    
    def is_fitted(self):
        return self._is_fitted
    
    
    def _most_common(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
        
        
    def _entropy(self, y):
        probas = (1/len(y)) * np.bincount(y)
        return np.sum([p * np.log2(1/p) for p in probas if p > 0])
    
    
    def _information_gain(self, X_column, y, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)
        # children entropy
        left_idxs, right_idxs = self._split(X_column, threshold)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0: # Important
            return 0
        
        left_entropy = self._entropy(y[left_idxs])
        right_entropy = self._entropy(y[right_idxs])
        n_samples = len(y)
        n_left_samples = len(left_idxs)
        n_right_samples = len(right_idxs)
        children_entropy = (1/n_samples) * (n_left_samples * left_entropy + n_right_samples * right_entropy)
        return parent_entropy - children_entropy


    def _build_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        
        if (depth > self.max_depth or n_samples < self.min_samples_split or n_labels == 1):
            leaf_value = self._most_common(y)
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(self.n_features, n_feats, replace=False)
        
        best_feat_idx, best_threshold = self._best_split(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat_idx], best_threshold)
        
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth=depth+1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth=depth+1)
        return Node(best_feat_idx, best_threshold, left, right)
    
    
    def _split(self, X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs
        
    
    def _best_split(self, X, y, feat_idxs):
        best_feat_idx = None
        best_threshold = None
        best_gain = -1
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(X_column, y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feat_idx = feat_idx
                    best_threshold = threshold
                    
        return best_feat_idx, best_threshold
            
            
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        
        if self.random_state:
            np.random.seed(self.random_state)
            
        self.root = self._build_tree(X, y)
        self._is_fitted = True
        return self
    

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])


    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        feat_value = x[node.feature]
        next_node = node.right if feat_value > node.threshold else node.left
        return self._traverse_tree(x, next_node)



if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf.fit(X_train, y_train)
    
    print("Train score", accuracy_score(y_train, clf.predict(X_train))) 
    print("Test score", accuracy_score(y_test, clf.predict(X_test)))   
        
        
