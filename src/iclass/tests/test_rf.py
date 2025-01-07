"""Tests for the routines of the RF module (rf_func).
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
from iclass.rf import feature_importance, train_rf, apply_rf

class TestFeatureImportance(unittest.TestCase):

    def test_feature_importance(self):
        # Define mock feature names and importances
        feature_names = ['feature1', 'feature2', 'feature3']
        importances = [0.3, 0.5, 0.2]

        # Create a mock classifier with predefined feature importances
        mock_clf = MagicMock()
        mock_clf.feature_importances_ = importances

        # Call the function with the mock classifier
        result = feature_importance(feature_names, mock_clf)

        # Define expected output
        expected_df = pd.DataFrame({
            'Feature': ['feature2', 'feature1', 'feature3'],
            'Importance': [0.5, 0.3, 0.2]
        }).reset_index(drop=True)

        # Check if the result matches the expected DataFrame
        pd.testing.assert_frame_equal(result.reset_index(drop=True),
                                      expected_df)

class TestTrainRF(unittest.TestCase):

    @patch('iclass.rf.RandomForestClassifier')
    def test_train_rf_with_config(self, MockRF):
        # Mock configuration and training data
        config = {
            'random_forest_args': {'n_estimators': 10, 'max_depth': 5},
            'random_forest_features': ['feature1', 'feature2']
        }
        df_train = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'psf_class': [0, 1, 0]
        })

        # Mock the RandomForestClassifier instance
        mock_clf = MockRF.return_value

        # Run the train_rf function with config
        clf = train_rf(df_train, config)

        # Assert RandomForestClassifier was initialized with the given 
        # arguments
        MockRF.assert_called_once_with(**config['random_forest_args'])

        # Verify clf.fit was called with the specified features and target
        actual_args = mock_clf.fit.call_args[0]
        expected_features = df_train[['feature1', 'feature2']].values
        expected_target = df_train['psf_class'].values

        self.assertEqual(actual_args[0].shape, expected_features.shape,
                         "Feature shapes do not match")
        self.assertTrue((actual_args[0].values == expected_features).all(),
                        "Feature values do not match")
        self.assertEqual(actual_args[1].shape, expected_target.shape,
                         "Target shapes do not match")
        self.assertTrue((actual_args[1].values == expected_target).all(),
                        "Target values do not match")



    @patch('iclass.rf.RandomForestClassifier')
    def test_train_rf_without_config(self, MockRF):
        # Mock training data with additional features
        df_train = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'psf_class': [0, 1, 0]
        })

        # Mock the RandomForestClassifier instance
        mock_clf = MockRF.return_value

        # Run the train_rf function without a config
        clf = train_rf(df_train, config=None)

        # Assert RandomForestClassifier was initialized with default arguments
        MockRF.assert_called_once_with()

        # Verify clf.fit was called with the specified features and target
        actual_args = mock_clf.fit.call_args[0]  
        expected_features = df_train[['feature1', 'feature2']].values
        expected_target = df_train['psf_class'].values

        self.assertEqual(actual_args[0].shape, expected_features.shape,
                         "Feature shapes do not match")
        self.assertTrue((actual_args[0].values == expected_features).all(),
                        "Feature values do not match")
        self.assertEqual(actual_args[1].shape, expected_target.shape,
                         "Target shapes do not match")
        self.assertTrue((actual_args[1].values == expected_target).all(),
                        "Target values do not match")

    def test_apply_rf(self):
        rf = Mock()
        rf.feature_names_in_ = ['x', 'y']
        rf.predict = Mock(return_value = [1, 2, 3])

        data = dict(
            x = [0, 0, 0],
            y = [1, 1, 1],
        )
        sample = pd.DataFrame(data=data)
        result = apply_rf(sample, rf)

        self.assertListEqual(
            result['reco_psf_class'].to_list(),
            rf.predict.return_value
        )
