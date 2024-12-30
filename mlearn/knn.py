import numpy as np
from collections import Counter

class KNNClassifier:
    """
    A simple K-Nearest Neighbors (KNN) classifier.
    """
    def __init__(self, n_neighbors=5):
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")
        self.n_neighbors = n_neighbors
        self._is_fitted = False
    
    def is_fitted(self):
        return self._is_fitted
    
    def fit(self, X, y):
        """
        Fit the classifier by storing the training data and labels.
        """
        if X.shape[0] != len(y):
            raise ValueError("Number of samples in X and y must be equal.")
        
        self.X = np.array(X)
        self.y = np.array(y)
        self._is_fitted = True
    
    def _compute_distance(self, x1, x2):
        """
        Compute the squared Euclidean distance between two points.
        """
        return np.linalg.norm(x1 - x2)
    
    def _find_nearest_neighbors(self, v):
        """
        Find the most common class among the nearest neighbors of a point.
        """
        distances = np.array([self._compute_distance(v, x) for x in self.X])
        nn_idxs = distances.argsort()[:self.n_neighbors]
        nearest_labels = self.y[nn_idxs]
        counter = Counter(nearest_labels)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        """
        Predict the class labels for the input samples.
        """
        if not self._is_fitted:
            raise ValueError("The classifier has not been fitted yet.")
        
        X = np.array(X)
        if X.shape[1] != self.X.shape[1]:
            raise ValueError("Number of features in X must match the training data.")
        
        return [self._find_nearest_neighbors(x) for x in X]

#######################################################################

class KNNRegressor:
    """
    A simple K-Nearest Neighbors (KNN) regressor.
    """
    def __init__(self, n_neighbors=5):
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")
        self.n_neighbors = n_neighbors
        self._is_fitted = False
    
    def is_fitted(self):
        return self._is_fitted
    
    def fit(self, X, y):
        """
        Fit the regressor by storing the training data and target.
        """
        if X.shape[0] != len(y):
            raise ValueError("Number of samples in X and y must be equal.")
        
        self.X = np.array(X)
        self.y = np.array(y)
        self._is_fitted = True
    
    def _compute_distance(self, x1, x2):
        """
        Compute the squared Euclidean distance between two points.
        """
        return np.linalg.norm(x1 - x2)
    
    def _find_nearest_neighbors(self, v):
        """
        Find the most common class among the nearest neighbors of a point.
        """
        distances = np.array([self._compute_distance(v, x) for x in self.X])
        nn_idxs = distances.argsort()[:self.n_neighbors]
        nearest_labels = self.y[nn_idxs]
        return nearest_labels.mean()
    
    def predict(self, X):
        """
        Predict the class labels for the input samples.
        """
        if not self._is_fitted:
            raise ValueError("The classifier has not been fitted yet.")
        
        X = np.array(X)
        if X.shape[1] != self.X.shape[1]:
            raise ValueError("Number of features in X must match the training data.")
        
        return [self._find_nearest_neighbors(x) for x in X]


##########################################################################



if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load the breast cancer dataset
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the classifier
    clf = KNNClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    
    # Evaluate the classifier
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.3f}")
    print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train):.3f}")

