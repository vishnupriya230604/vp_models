import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Linear_regression import Linear_regression


import numpy as np
from models.Linear_regression import Linear_regression

def test_linear_regression():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    model = Linear_regression()
    model.fit(X, y)
    predictions = model.predict(X)
    print("Predictions:", predictions)
    print("Actual:", y)

    assert np.allclose(predictions, y, atol=1e-2), "Predictions are not within acceptable tolerance"


if __name__ == "__main__":
    test_linear_regression()
    print("All tests passed for Linear_regression")