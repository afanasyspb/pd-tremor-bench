"""
Stochastic Configuration Network (SCN) for binary tremor-state classification.

The model is a randomized single-hidden-layer network: hidden weights and biases
are drawn from a fixed distribution and the output weights are solved in closed
form with ridge regression. Several independent networks are trained and their
probabilities are averaged, which yields both a point prediction and an
ensemble-based uncertainty estimate (the per-sample standard deviation).

Reproducibility
---------------
Each run uses its own local `numpy.random.RandomState(random_state + run_id)`.
This makes the ensemble deterministic for a given `random_state` and, unlike the
global `numpy.random.seed`, never mutates NumPy's global RNG, so the model cannot
perturb (or be perturbed by) any other component that shares that global state.

Parameters
----------
num_hidden : int, default=100
    Number of hidden neurons per network.
num_runs : int, default=50
    Number of independent networks in the ensemble (clamped to a minimum of 30
    so the uncertainty estimate is stable).
random_state : int, default=42
    Base seed; run ``i`` uses ``random_state + i``.
activation : {'relu', 'sigmoid', 'tanh'}, default='relu'
    Hidden-layer activation function.
regularization : float, default=0.01
    L2 (ridge) penalty on the closed-form output-weight solve.
verbose : bool, default=True
    If True, print initialization and training progress.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class StochasticConfigurationNetwork:
    """Randomized single-hidden-layer ensemble with ridge-solved output weights."""

    def __init__(self, num_hidden=100, num_runs=50, random_state=42,
                 activation='relu', regularization=0.01, verbose=True):
        self.num_hidden = num_hidden
        self.num_runs = max(30, num_runs)  # ensemble needs >= 30 runs to be stable
        self.random_state = random_state
        self.activation = activation
        self.regularization = regularization
        self.verbose = verbose

        # Per-run parameters, populated by fit().
        self.weights_hidden = []
        self.bias_hidden = []
        self.weights_output = []
        self.scaler = StandardScaler()

        if self.verbose:
            print("Stochastic Configuration Network initialized "
                  f"(hidden={self.num_hidden}, runs={self.num_runs}, "
                  f"activation={self.activation}, lambda={self.regularization})")

    def _activation_fn(self, x):
        """Apply the configured element-wise activation function."""
        if self.activation == 'relu':
            return np.maximum(0, x)
        if self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        if self.activation == 'tanh':
            return np.tanh(x)
        raise ValueError(f"Unknown activation: {self.activation}")

    def _single_run(self, X_train, y_train, run_id):
        """Train one randomized network and return its parameters.

        A local RandomState seeded with ``random_state + run_id`` draws the
        hidden weights and biases, so the run is reproducible without touching
        the global NumPy RNG.
        """
        rng = np.random.RandomState(self.random_state + run_id)
        n_features = X_train.shape[1]

        # Randomly configured hidden layer (fixed, not trained).
        weights_h = rng.randn(n_features, self.num_hidden) * 0.5
        bias_h = rng.randn(1, self.num_hidden) * 0.5

        H = self._activation_fn(X_train @ weights_h + bias_h)
        H_aug = np.hstack([H, np.ones((H.shape[0], 1))])  # append bias column

        # Closed-form ridge solution for the output weights:
        # (H^T H + lambda I) w = H^T y
        HTH = H_aug.T @ H_aug
        HTy = H_aug.T @ y_train.reshape(-1, 1)
        lambda_matrix = self.regularization * np.eye(HTH.shape[0])
        weights_out = np.linalg.solve(HTH + lambda_matrix, HTy)

        return weights_h, bias_h, weights_out

    def fit(self, X_train, y_train):
        """Fit the ensemble on training data.

        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
        y_train : array-like, shape (n_samples,)
            Binary labels (0 or 1).
        """
        X_train = self.scaler.fit_transform(X_train)

        if self.verbose:
            print(f"Training SCN ensemble: {self.num_runs} runs "
                  f"on data of shape {X_train.shape}")

        for run_id in range(self.num_runs):
            w_h, b_h, w_o = self._single_run(X_train, y_train, run_id)
            self.weights_hidden.append(w_h)
            self.bias_hidden.append(b_h)
            self.weights_output.append(w_o)

        if self.verbose:
            print("SCN training complete")

        return self

    def _predict_single_run(self, X_test_norm, run_id):
        """Return the positive-class probability from a single network."""
        H = self._activation_fn(
            X_test_norm @ self.weights_hidden[run_id] + self.bias_hidden[run_id])
        H_aug = np.hstack([H, np.ones((H.shape[0], 1))])
        logits = H_aug @ self.weights_output[run_id]
        proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        return proba.flatten()

    def predict_proba(self, X_test):
        """Return the ensemble mean probability and its uncertainty.

        Returns
        -------
        mean_proba : array, shape (n_samples,)
            Mean positive-class probability across all runs.
        std_proba : array, shape (n_samples,)
            Per-sample standard deviation across runs (ensemble uncertainty).
        """
        X_test_norm = self.scaler.transform(X_test)
        probas = np.array([self._predict_single_run(X_test_norm, r)
                           for r in range(self.num_runs)])
        return probas.mean(axis=0), probas.std(axis=0)

    def predict(self, X_test, threshold=0.5):
        """Return binary predictions from the ensemble mean probability."""
        mean_proba, _ = self.predict_proba(X_test)
        return (mean_proba >= threshold).astype(int)

    def predict_with_uncertainty(self, X_test, threshold=0.5):
        """Return predictions together with the ensemble uncertainty.

        Returns
        -------
        predictions : array, shape (n_samples,)
        std_proba : array, shape (n_samples,)
            Ensemble standard deviation (uncertainty) per sample.
        mean_proba : array, shape (n_samples,)
            Ensemble mean probability per sample.
        """
        mean_proba, std_proba = self.predict_proba(X_test)
        predictions = (mean_proba >= threshold).astype(int)
        return predictions, std_proba, mean_proba
