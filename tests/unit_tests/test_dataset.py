import unittest
import numpy as np
from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())

    def test_dropna(self):
        X = np.array([[1, 2, 3], [np.nan, 5, 6], [7, 8, 9], [10, 11, np.nan]])
        y = np.array([1, 2, 3, 4])
        dataset = Dataset(X, y, ['a', 'b', 'c'], 'y')
 
        result = dataset.dropna()
 
        self.assertIs(result, dataset)
        self.assertEqual((2, 3), dataset.shape())
        self.assertFalse(np.isnan(dataset.X).any())
        np.testing.assert_array_equal(np.array([[1, 2, 3], [7, 8, 9]]), dataset.X)
        np.testing.assert_array_equal(np.array([1, 3]), dataset.y)
 
    def test_dropna_without_nan(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])
        dataset = Dataset(X, y)
 
        dataset.dropna()
 
        self.assertEqual((2, 2), dataset.shape())
        np.testing.assert_array_equal(np.array([0, 1]), dataset.y)
 
    def test_dropna_without_label(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        dataset = Dataset(X)
 
        dataset.dropna()
 
        self.assertEqual((1, 2), dataset.shape())
        self.assertFalse(dataset.has_label())
 
    def test_fillna_with_value(self):
        X = np.array([[1.0, np.nan], [np.nan, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 0])
        dataset = Dataset(X, y)
 
        result = dataset.fillna(0)
 
        self.assertIs(result, dataset)
        self.assertFalse(np.isnan(dataset.X).any())
        np.testing.assert_array_equal(np.array([[1.0, 0.0], [0.0, 4.0], [5.0, 6.0]]), dataset.X)
        np.testing.assert_array_equal(np.array([0, 1, 0]), dataset.y)
 
    def test_fillna_with_mean(self):
        X = np.array([[1.0, 10.0], [np.nan, 20.0], [2.0, np.nan], [9.0, 40.0]])
        dataset = Dataset(X)
 
        dataset.fillna("mean")
 
        self.assertFalse(np.isnan(dataset.X).any())
        self.assertAlmostEqual(4.0, dataset.X[1, 0])
        self.assertAlmostEqual(70 / 3, dataset.X[2, 1])
 
    def test_fillna_with_median(self):
        X = np.array([[1.0, 10.0], [np.nan, 20.0], [2.0, np.nan], [9.0, 40.0]])
        dataset = Dataset(X)
 
        dataset.fillna("median")
 
        self.assertFalse(np.isnan(dataset.X).any())
        self.assertAlmostEqual(2.0, dataset.X[1, 0])
        self.assertAlmostEqual(20.0, dataset.X[2, 1])
 
    def test_fillna_without_nan(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        dataset = Dataset(X)
 
        dataset.fillna("mean")
 
        np.testing.assert_array_equal(np.array([[1.0, 2.0], [3.0, 4.0]]), dataset.X)
 
    def test_remove_by_index(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([10, 20, 30])
        dataset = Dataset(X, y)
 
        result = dataset.remove_by_index(1)
 
        self.assertIs(result, dataset)
        self.assertEqual((2, 2), dataset.shape())
        np.testing.assert_array_equal(np.array([[1, 2], [5, 6]]), dataset.X)
        np.testing.assert_array_equal(np.array([10, 30]), dataset.y)
 
    def test_remove_by_index_without_label(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        dataset = Dataset(X)
 
        dataset.remove_by_index(0)
 
        self.assertEqual((2, 2), dataset.shape())
        np.testing.assert_array_equal(np.array([[3, 4], [5, 6]]), dataset.X)
        self.assertFalse(dataset.has_label())