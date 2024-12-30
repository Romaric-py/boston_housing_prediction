import json
import numpy as np

class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None
        self._is_fitted = False
    
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self._is_fitted = True

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("This scaler was not fitted before")
        
    def transform(self, X):
        self._check_is_fitted()
        copyX = X.copy()
        copyX = (copyX - self.mean) / self.std
        return copyX
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    
    def to_json(self, path):
        self._check_is_fitted()
        params = {'mean': self.mean.tolist(), 'std': self.std.tolist()}
        with open(path, 'w') as f:
            json.dump(params, f, indent=4)

    def load_from_json(self, path):
        with open(path, 'r') as f:
            params = json.load(f)
        self._is_fitted = True
        self.mean = np.array(params['mean'])
        self.std = np.array(params['std'])