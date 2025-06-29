from sklearn.datasets import load_iris

def test_iris_feature_count():
    iris = load_iris()
    x = iris.data
    assert x.shape[1] == 4, f"Expected 4 features, but got {x.shape[1]}"