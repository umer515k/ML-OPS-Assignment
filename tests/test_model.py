import os
import sys
import pathlib
import pytest

# make src importable if repository layout is different
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from model import train_model, predict, MODEL_PATH

def test_train_and_predict(tmp_path):
    # Ensure model trains and returns accuracy between 0 and 1
    acc = train_model()
    assert 0.0 <= acc <= 1.0

    # After training, predict should return 0 or 1
    p = predict(2, 7)  # adapt args to your model signature
    assert p in (0, 1)


