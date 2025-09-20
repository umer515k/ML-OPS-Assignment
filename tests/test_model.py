import numpy as np
from sklearn.linear_model import LinearRegression

def test_model_prediction():
    # Train a tiny dummy model
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    y = np.array([10, 20, 30])
    model = LinearRegression().fit(X, y)

    # Make a prediction
    pred = model.predict([[4, 4, 4]])
    assert pred[0] > 0
