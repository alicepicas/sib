import unittest
import numpy as np
from si.data.dataset import Dataset
from si.feature_selection.select_percentile import SelectPercentile
 
 
class TestSelectPercentile(unittest.TestCase):
 
    def setUp(self):
        self.X = np.array([[0.0, 1.0, 5.0, 2.0],
                           [0.1, 3.0, 5.2, 1.0],
                           [5.0, 2.0, 5.1, 2.5],
                           [5.1, 3.0, 4.9, 1.5],
                           [10.0, 1.5, 5.3, 2.2],
                           [10.1, 2.5, 4.8, 1.8]])
        self.y = np.array([0, 0, 1, 1, 2, 2])
        self.features = ['a', 'b', 'c', 'd']
        self.dataset = Dataset(self.X, self.y, self.features, 'cls')
 
    def test_fit(self):
        selector = SelectPercentile(percentile=50)
        selector.fit(self.dataset)
 
        self.assertEqual((4,), selector.F.shape)
        self.assertEqual((4,), selector.p.shape)
        self.assertTrue(np.all((selector.p >= 0) & (selector.p <= 1)))
        self.assertEqual(0, np.argmax(selector.F))
 
    def test_transform_selects_best_features(self):
        selector = SelectPercentile(percentile=25)
        transformed = selector.fit_transform(self.dataset)
 
        self.assertEqual((6, 1), transformed.shape())
        self.assertEqual(['a'], transformed.features)
        np.testing.assert_array_equal(self.X[:, [0]], transformed.X)
 
    def test_transform_keeps_label_and_original_dataset(self):
        selector = SelectPercentile(percentile=50)
        transformed = selector.fit_transform(self.dataset)
 
        np.testing.assert_array_equal(self.y, transformed.y)
        self.assertEqual('cls', transformed.label)
        self.assertEqual((6, 4), self.dataset.shape())
 
    def test_transform_keeps_original_feature_order(self):
        F = np.array([1.0, 4.0, 2.0, 3.0])
        selector = SelectPercentile(score_func=lambda dataset: (F, None), percentile=50)
        transformed = selector.fit_transform(self.dataset)
 
        self.assertEqual(['b', 'd'], transformed.features)
        np.testing.assert_array_equal(self.X[:, [1, 3]], transformed.X)
 
    def test_transform_with_ties(self):
        F = np.array([1.2, 3.4, 2.1, 5.6, 4.3, 5.6, 7.8, 6.5, 5.6, 3.2])
        X = np.random.rand(5, 10)
        features = [f"f{i}" for i in range(10)]
        dataset = Dataset(X, np.array([0, 1, 0, 1, 0]), features, 'y')
 
        selector = SelectPercentile(score_func=lambda dataset: (F, None), percentile=40)
        transformed = selector.fit_transform(dataset)
 
        self.assertEqual((5, 4), transformed.shape())
        self.assertEqual(['f3', 'f5', 'f6', 'f7'], transformed.features)
        np.testing.assert_array_equal(X[:, [3, 5, 6, 7]], transformed.X)
 
    def test_transform_with_full_percentile(self):
        selector = SelectPercentile(percentile=100)
        transformed = selector.fit_transform(self.dataset)
 
        self.assertEqual((6, 4), transformed.shape())
        self.assertEqual(self.features, transformed.features)
