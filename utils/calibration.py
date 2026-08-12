import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class TemperatureScaling:
    """Temperature Scaling for multi-class calibration.

    Scales logits by dividing by temperature T before softmax:
        p_calibrated = softmax(logits / T)

    Optimal T found by minimizing negative log-likelihood on validation set.
    """

    def __init__(self):
        self.temperature = 1.0
        self.fitted = False

    def fit(
        self, logits: np.ndarray, labels: np.ndarray, verbose: bool = True
    ) -> "TemperatureScaling":
        """
        Find optimal temperature on validation set.

        Args:
            logits: Raw model outputs (N, num_classes), pre-softmax
            labels: True class indices (N,)
            verbose: Print optimization results

        Returns:
            self (fitted)
        """
        # Convert labels to int if needed
        if labels.dtype != np.int64 and labels.dtype != np.int32:
            labels = labels.astype(np.int64)

        # Negative log-likelihood loss
        def nll_loss(T):
            T = T[0]  # scipy passes scalar as array
            if T <= 0:
                return 1e10  # Invalid temperature
            scaled_logits = logits / T
            probs = softmax(scaled_logits, axis=1)
            # Add small epsilon to avoid log(0)
            probs = np.clip(probs, 1e-10, 1.0)
            nll = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
            return nll

        # Optimize
        result = minimize(
            nll_loss,
            x0=[1.0],  # Start with T=1 (no scaling)
            method="Nelder-Mead",
            options={"maxiter": 1000},
        )

        self.temperature = result.x[0]
        self.fitted = True

        if verbose:
            print("Temperature Scaling fitted:")
            print(f"  Optimal Temperature: {self.temperature:.4f}")
            print(f"  NLL (before): {nll_loss([1.0]):.4f}")
            print(f"  NLL (after): {result.fun:.4f}")

            if self.temperature < 1.0:
                print("\tModel is UNDERCONFIDENT (T<1), sharpening predictions")
            elif self.temperature > 1.0:
                print("\tModel is OVERCONFIDENT (T>1), smoothing predictions")
            else:
                print("\tModel is well-calibrated (T≈1)")

        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Raw model outputs (N, num_classes)

        Returns:
            Calibrated probabilities (N, num_classes)
        """
        if not self.fitted:
            warnings.warn("TemperatureScaling not fitted, using T=1 (no scaling)")
            return softmax(logits, axis=1)

        scaled_logits = logits / self.temperature
        return softmax(scaled_logits, axis=1)

    def fit_transform(
        self, logits: np.ndarray, labels: np.ndarray, verbose: bool = True
    ) -> np.ndarray:
        """Fit and transform in one call."""
        self.fit(logits, labels, verbose)
        return self.transform(logits)


class PlattScaling:
    """Platt Scaling (logistic regression calibration).

    Fits one-vs-rest logistic regression: p_calibrated = sigmoid(A * logit + B).
    More flexible than temperature scaling (2 parameters per class).
    """

    def __init__(self):
        self.models = {}  # One binary classifier per class (one-vs-rest)
        self.fitted = False

    def fit(
        self, logits: np.ndarray, labels: np.ndarray, verbose: bool = True
    ) -> "PlattScaling":
        """
        Fit logistic regression calibrators (one-vs-rest).

        Args:
            logits: Raw model outputs (N, num_classes)
            labels: True class indices (N,)
            verbose: Print training info
        """
        num_classes = logits.shape[1]

        for c in range(num_classes):
            # Binary labels for this class
            y_binary = (labels == c).astype(int)

            # Use logit for this class as feature
            X = logits[:, c].reshape(-1, 1)

            # Fit logistic regression
            model = LogisticRegression(solver="lbfgs", max_iter=1000)
            model.fit(X, y_binary)

            self.models[c] = model

        self.fitted = True

        if verbose:
            print(f"Platt Scaling fitted for {num_classes} classes")

        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling.

        Args:
            logits: Raw model outputs (N, num_classes)

        Returns:
            Calibrated probabilities (N, num_classes)
        """
        if not self.fitted:
            raise ValueError("PlattScaling not fitted. Call fit() first.")

        num_classes = logits.shape[1]
        calibrated_probs = np.zeros_like(logits)

        for c in range(num_classes):
            X = logits[:, c].reshape(-1, 1)
            calibrated_probs[:, c] = self.models[c].predict_proba(X)[:, 1]

        # Normalize to sum to 1
        calibrated_probs = calibrated_probs / calibrated_probs.sum(
            axis=1, keepdims=True
        )

        return calibrated_probs


class IsotonicCalibration:
    """Isotonic Regression calibration.

    Non-parametric method that learns a monotonic mapping from
    predicted probabilities to calibrated probabilities.
    """

    def __init__(self):
        self.calibrators = {}
        self.fitted = False

    def fit(
        self, probs: np.ndarray, labels: np.ndarray, verbose: bool = True
    ) -> "IsotonicCalibration":
        """
        Fit isotonic regression (one-vs-rest).

        Args:
            probs: Predicted probabilities (N, num_classes), NOT logits
            labels: True class indices (N,)
            verbose: Print training info
        """
        num_classes = probs.shape[1]

        for c in range(num_classes):
            y_binary = (labels == c).astype(int)

            # Fit isotonic regression
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(probs[:, c], y_binary)

            self.calibrators[c] = calibrator

        self.fitted = True

        if verbose:
            print(f"Isotonic Calibration fitted for {num_classes} classes")

        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration.

        Args:
            probs: Predicted probabilities (N, num_classes)

        Returns:
            Calibrated probabilities (N, num_classes)
        """
        if not self.fitted:
            raise ValueError("IsotonicCalibration not fitted. Call fit() first.")

        num_classes = probs.shape[1]
        calibrated_probs = np.zeros_like(probs)

        for c in range(num_classes):
            calibrated_probs[:, c] = self.calibrators[c].predict(probs[:, c])

        # Normalize to sum to 1
        calibrated_probs = calibrated_probs / calibrated_probs.sum(
            axis=1, keepdims=True
        )

        return calibrated_probs
