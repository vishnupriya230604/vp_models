import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Multivariate_regression import Multivariate_regression


import numpy as np

def test_multiple_regression():
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([1, 2, 3])
    model = Multivariate_regression()
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, y)

if __name__ == "__main__":
    test_multiple_regression()
    print("All tests passed for My_Multiple_Regression")