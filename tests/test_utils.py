from src.utils import preprocess

def test_preprocess():
    x = [0]*784
    result = preprocess(x)
    assert len(result) == 784
