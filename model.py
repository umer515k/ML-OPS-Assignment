import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

# Paths
DATASET_PATH = os.path.join(os.getcwd(), "datasets", "hiring.csv")
MODEL_PATH = os.path.join(os.getcwd(), "src", "model.pkl")


def train_model():
    """Train the regression model and save it to disk."""
    dataset = pd.read_csv(DATASET_PATH)

    X = dataset.iloc[:, :3]
    y = dataset.iloc[:, -1]

    regressor = LinearRegression()
    regressor.fit(X, y)

    # Save trained model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(regressor, f)

    return regressor


def load_model():
    """Load the saved model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not trained yet. Run train_model().")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


def predict(features):
    """
    Predict using the trained model.
    Args:
        features (list or tuple): e.g. [2, 9, 6]
    Returns:
        float: prediction result
    """
    model = load_model()
    return model.predict([features])[0]
