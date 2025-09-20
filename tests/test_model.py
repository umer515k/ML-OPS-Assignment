import pickle
import numpy as np

# Load the model from pickle
model = pickle.load(open("model.pkl", "rb"))

def test_prediction_shape():
    """Check if model returns prediction with correct shape"""
    features = np.array([[2, 9, 6]])
    prediction = model.predict(features)
    assert prediction.shape == (1,)

def test_prediction_type():
    """Check if prediction is a float or int"""
    features = np.array([[2, 9, 6]])
    prediction = model.predict(features)
    assert isinstance(prediction[0].item(), (float, int))
