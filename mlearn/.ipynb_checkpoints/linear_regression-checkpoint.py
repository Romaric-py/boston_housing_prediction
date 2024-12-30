import numpy as np
import matplotlib.pyplot as plt
import json
from dataclasses import dataclass


@dataclass
class LinearRegression:
    learning_rate: float = 0.01
    precision: float = 0.01
    n_iter: int = 1000
    n_iter_no_change: int = 10
    fit_intercept: bool = True
    
    def __post_init__(self):
        self._is_fitted = False
        self._n_no_change = 0
        self.coef_ = None
        self.intercept_ = None
        self.history = {'loss': []}

    def _init_params(self, X):
        """Initialise les poids du modèle en fonction de la taille de X"""
        n = X.shape[1]
        W = np.random.randn(n, 1)
        b = np.random.randn()
        return W, b

    def _output(self, X, W, b):
        """Calcule la sortie du modèle avec les paramètres fournis"""
        return X.dot(W) + b

    def _error(self, y_pred, y_true):
        """Ecarts (non absolu) entre prédictions et valeurs réelles"""
        return y_pred - y_true

    def _loss(self, error):
        """Calcule la MSE"""
        return 0.5 * np.mean(error**2, axis=0)

    def _compute_gradient(self, X, error):
        """Calcule le gradient de la fonction de perte"""
        m = X.shape[0]
        gradW = (1 / m) * (X.T.dot(error))
        partial_b = np.mean(error, axis=0)
        return gradW, partial_b

    def _check_data_shape(self, X, y):
        if len(X.shape) != 2:
            raise ValueError("X must be 2D array")
        elif len(y.shape) == 1:
            if y.shape[0] != X.shape[0]:
                raise ValueError(f"X shape '{X.shape}' is incompatible with y shape '{y.shape}'")
            else:
                y = y.reshape(-1, 1)
        elif len(y.shape) == 2:
            if y.shape[1] != 1:
                raise ValueError(f"Invalid shape for y; expected: ({X.shape[0]},) or ({X.shape[0]}, 1); ; found {y.shape}")
            elif y.shape[0] != X.shape[0]:
                raise ValueError(f"X shape '{X.shape}' is incompatible with y shape '{y.shape}'")
        else:
            raise ValueError(f"Invalid shape for y; expected: (m,) or (m, 1); found {y.shape}")
        return X, y

    def _stop_training(self):
        loss = self.history['loss'][-1]
        last_loss = (self.history['loss'][-2] if len(self.history['loss']) > 1 else 0)
        if abs(loss - last_loss) < self.precision:
            self._n_no_change += 1
        else:
            self._n_no_change = 0
        return (self._n_no_change == self.n_iter_no_change)

    def _check_is_fitted(self):
        if not self.is_fitted():
            raise RuntimeError("This LinearRegression instance is not fitted yet. Call 'fit' "
                               "with appropriate arguments before using this estimator.")    

    def fit(self, X, y, verbose=False):
        """Entraînement du modèle"""
        # Convertir en matrices numpy
        X = np.array(X)
        y = np.array(y)
        # Vérifier la forme des données
        X, y = self._check_data_shape(X, y)
        # Initialiser les paramètres du modèle
        W, b = self._init_params(X)
        b = (b if self.fit_intercept else np.zeros((1,)))

        y_pred = self._output(X, W, b) # Calcul de la sortie
        error = self._error(y_pred, y) # Matrice des écarts
        # Boucle d'entraînement
        for i in range(self.n_iter):
            # Mise à jour des paramètres
            gradW, partial_b = self._compute_gradient(X, error)
            W -= self.learning_rate * gradW
            b -= (self.learning_rate * partial_b if self.fit_intercept else np.zeros((1,)))
            
            y_pred = self._output(X, W, b) # Calcul de la sortie
            error = self._error(y_pred, y) # Matrice des écarts
            
            loss = self._loss(error)  # Calcul du coût
            self.history['loss'].append(loss) # Enregistrer la valeur du coût
            if verbose:
                print(f"Iteration n°{i+1} - Loss {loss}")
            # Arreter la boucle si le modèle n'apprend plus
            if self._stop_training():
                if verbose:
                    print(f"Early stopping after {self.n_iter_no_change} iterations without significant change")
                break

        # Fin de l'entraînement
        self.coef_ = W
        self.intercept_ = b
        self._is_fitted = True

    def is_fitted(self):
        return self._is_fitted
    
    def plot(self):
        self._check_is_fitted()
        if self.history['loss']:
            plt.title("Loss Curve")
            plt.xlabel("Epochs")
            plt.ylabel("Loss (MSE)")
            return plt.plot(self.history['loss'])

    def predict(self, X):
        self._check_is_fitted()
        return self._output(X, self.coef_, self.intercept_)

    def to_json(self, path):
        self._check_is_fitted()
        params = {'coef_': self.coef_.tolist(), 'intercept_': self.intercept_.tolist()}
        with open(path, 'w') as f:
            json.dump(params, f, indent=4)

    def load_from_json(self, path):
        with open(path, 'r') as f:
            params = json.load(f)
        self._is_fitted = True
        self.coef_ = np.array(params['coef_'])
        self.intercept_ = np.array(params['intercept_'])
        
        



            

        

            

        
            
        
        
        