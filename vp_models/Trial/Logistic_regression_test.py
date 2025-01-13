import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Logistic_regression import Logistic_regression


import numpy as np


def test_logistic_regression():
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([0, 1, 0])
    model = Logistic_regression()
    model.fit(X, y)
    predictions = model.predict(X)
    print("Predictions:", predictions)
    print("Actual:", y)
    assert np.allclose(predictions, y, atol=1e-2), "Predictions are not within acceptable tolerance"

if __name__ == "__main__":
    test_logistic_regression()
    print("All tests passed for Logistic_regression")