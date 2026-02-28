"""Test suite for cover song identification system."""

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.synthetic_dataset import SyntheticCoverSongDataset
from src.features.audio_features import AudioFeatureExtractor
from src.metrics.evaluation import CoverSongMetrics
from src.models.baseline import BaselineCoverSongIdentifier
from src.models.dtw_crp import CrossRecurrencePlot, DynamicTimeWarping
from src.utils.device import get_device, set_seed


class TestAudioFeatureExtractor(unittest.TestCase):
    """Test audio feature extraction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = AudioFeatureExtractor()
        self.sample_audio = np.random.randn(22050)  # 1 second of audio
    
    def test_feature_extraction(self):
        """Test basic feature extraction."""
        features = self.extractor.extract_features(self.sample_audio)
        
        # Check that all expected features are present
        expected_features = [
            "mfcc", "chroma", "spectral_contrast", "tempo", "rhythm",
            "zcr", "spectral_centroid", "spectral_rolloff"
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], np.ndarray)
            self.assertGreater(len(features[feature]), 0)
    
    def test_combined_features(self):
        """Test combined feature extraction."""
        combined_features = self.extractor.extract_combined_features(self.sample_audio)
        
        self.assertIsInstance(combined_features, np.ndarray)
        self.assertEqual(len(combined_features), self.extractor.get_feature_dimension())
    
    def test_empty_audio(self):
        """Test handling of empty audio."""
        empty_audio = np.array([])
        features = self.extractor.extract_features(empty_audio)
        
        # Should return zero features
        for feature_name, feature_values in features.items():
            self.assertTrue(np.all(feature_values == 0))


class TestDynamicTimeWarping(unittest.TestCase):
    """Test Dynamic Time Warping implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dtw = DynamicTimeWarping()
        self.seq1 = np.array([[1, 2], [3, 4], [5, 6]])
        self.seq2 = np.array([[1, 2], [3, 4], [5, 6]])
    
    def test_dtw_distance(self):
        """Test DTW distance computation."""
        distance, path = self.dtw.compute_distance(self.seq1, self.seq2)
        
        self.assertIsInstance(distance, float)
        self.assertIsInstance(path, np.ndarray)
        self.assertGreaterEqual(distance, 0)
    
    def test_dtw_similarity(self):
        """Test DTW similarity computation."""
        similarity = self.dtw.compute_similarity(self.seq1, self.seq2)
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0)
        self.assertLessEqual(similarity, 1)
    
    def test_identical_sequences(self):
        """Test DTW with identical sequences."""
        similarity = self.dtw.compute_similarity(self.seq1, self.seq1)
        self.assertAlmostEqual(similarity, 1.0, places=2)


class TestCrossRecurrencePlot(unittest.TestCase):
    """Test Cross-Recurrence Plot implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.crp = CrossRecurrencePlot(threshold=0.1)
        self.seq1 = np.array([[1, 2], [3, 4], [5, 6]])
        self.seq2 = np.array([[1, 2], [3, 4], [5, 6]])
    
    def test_crp_computation(self):
        """Test CRP computation."""
        crp_matrix = self.crp.compute_crp(self.seq1, self.seq2)
        
        self.assertIsInstance(crp_matrix, np.ndarray)
        self.assertEqual(crp_matrix.shape, (len(self.seq1), len(self.seq2)))
        self.assertTrue(np.all(np.logical_or(crp_matrix == True, crp_matrix == False)))
    
    def test_recurrence_rate(self):
        """Test recurrence rate computation."""
        crp_matrix = self.crp.compute_crp(self.seq1, self.seq2)
        recurrence_rate = self.crp.compute_recurrence_rate(crp_matrix)
        
        self.assertIsInstance(recurrence_rate, float)
        self.assertGreaterEqual(recurrence_rate, 0)
        self.assertLessEqual(recurrence_rate, 1)
    
    def test_determinism(self):
        """Test determinism computation."""
        crp_matrix = self.crp.compute_crp(self.seq1, self.seq2)
        determinism = self.crp.compute_determinism(crp_matrix)
        
        self.assertIsInstance(determinism, float)
        self.assertGreaterEqual(determinism, 0)
        self.assertLessEqual(determinism, 1)


class TestBaselineCoverSongIdentifier(unittest.TestCase):
    """Test baseline cover song identifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.identifier = BaselineCoverSongIdentifier()
        
        # Create dummy training data
        self.X_train = np.random.randn(100, self.identifier.feature_dim)
        self.y_train = np.random.randint(0, 2, 100)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.identifier.classifier)
        self.assertIsNotNone(self.identifier.feature_extractor)
        self.assertFalse(self.identifier.is_trained)
    
    def test_model_training(self):
        """Test model training."""
        metrics = self.identifier.train(self.X_train, self.y_train)
        
        self.assertTrue(self.identifier.is_trained)
        self.assertIn("train_accuracy", metrics)
        self.assertGreaterEqual(metrics["train_accuracy"], 0)
        self.assertLessEqual(metrics["train_accuracy"], 1)
    
    def test_model_prediction(self):
        """Test model prediction."""
        # Train model first
        self.identifier.train(self.X_train, self.y_train)
        
        # Test prediction
        X_test = np.random.randn(10, self.identifier.feature_dim)
        predictions = self.identifier.predict(X_test)
        
        self.assertEqual(len(predictions), len(X_test))
        self.assertTrue(np.all(np.logical_or(predictions == 0, predictions == 1)))
    
    def test_model_prediction_proba(self):
        """Test model probability prediction."""
        # Train model first
        self.identifier.train(self.X_train, self.y_train)
        
        # Test probability prediction
        X_test = np.random.randn(10, self.identifier.feature_dim)
        probabilities = self.identifier.predict_proba(X_test)
        
        self.assertEqual(len(probabilities), len(X_test))
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))


class TestCoverSongMetrics(unittest.TestCase):
    """Test cover song evaluation metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = CoverSongMetrics(k_values=[1, 5, 10])
        
        # Create dummy similarity matrix and labels
        self.similarities = np.random.rand(10, 10)
        self.labels = np.random.randint(0, 2, (10, 10))
    
    def test_map_at_k(self):
        """Test mAP@k computation."""
        sorted_labels = np.array([1, 0, 1, 0, 1])
        
        map_1 = self.metrics._compute_map_at_k(sorted_labels, 1)
        map_5 = self.metrics._compute_map_at_k(sorted_labels, 5)
        
        self.assertIsInstance(map_1, float)
        self.assertIsInstance(map_5, float)
        self.assertGreaterEqual(map_1, 0)
        self.assertLessEqual(map_1, 1)
        self.assertGreaterEqual(map_5, 0)
        self.assertLessEqual(map_5, 1)
    
    def test_recall_at_k(self):
        """Test R@k computation."""
        sorted_labels = np.array([1, 0, 1, 0, 1])
        
        recall_1 = self.metrics._compute_recall_at_k(sorted_labels, 1)
        recall_5 = self.metrics._compute_recall_at_k(sorted_labels, 5)
        
        self.assertIsInstance(recall_1, float)
        self.assertIsInstance(recall_5, float)
        self.assertGreaterEqual(recall_1, 0)
        self.assertLessEqual(recall_1, 1)
        self.assertGreaterEqual(recall_5, 0)
        self.assertLessEqual(recall_5, 1)
    
    def test_all_metrics(self):
        """Test computation of all metrics."""
        all_metrics = self.metrics.compute_all_metrics(
            self.similarities, self.labels
        )
        
        # Check that expected metrics are present
        expected_metrics = ["map_at_k", "recall_at_k"]
        for metric in expected_metrics:
            self.assertIn(metric, all_metrics)
    
    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        similarities_flat = self.similarities.flatten()
        labels_flat = self.labels.flatten()
        
        confusion_metrics = self.metrics.compute_confusion_matrix(
            similarities_flat, labels_flat, threshold=0.5
        )
        
        # Check that expected metrics are present
        expected_metrics = ["accuracy", "precision", "recall", "f1_score"]
        for metric in expected_metrics:
            self.assertIn(metric, confusion_metrics)
            self.assertGreaterEqual(confusion_metrics[metric], 0)
            self.assertLessEqual(confusion_metrics[metric], 1)


class TestSyntheticDataset(unittest.TestCase):
    """Test synthetic dataset generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset = SyntheticCoverSongDataset(
            data_dir=self.temp_dir,
            num_songs=5,
            num_covers_per_song=2,
            audio_duration=1.0,  # Short duration for testing
            privacy_mode=True
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_generation(self):
        """Test synthetic dataset generation."""
        df = self.dataset.generate_dataset()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        
        # Check that we have both original and cover songs
        original_count = len(df[df["label"] == "original"])
        cover_count = len(df[df["label"] == "cover"])
        
        self.assertGreater(original_count, 0)
        self.assertGreater(cover_count, 0)
    
    def test_audio_file_generation(self):
        """Test that audio files are actually generated."""
        df = self.dataset.generate_dataset()
        
        # Check that audio files exist
        for _, row in df.iterrows():
            self.assertTrue(os.path.exists(row["path"]))
    
    def test_cover_pairs(self):
        """Test cover pair generation."""
        df = self.dataset.generate_dataset()
        pairs = self.dataset.get_cover_pairs()
        
        self.assertIsInstance(pairs, list)
        self.assertGreater(len(pairs), 0)
        
        # Check that pairs are tuples of file paths
        for pair in pairs:
            self.assertIsInstance(pair, tuple)
            self.assertEqual(len(pair), 2)
            self.assertTrue(os.path.exists(pair[0]))
            self.assertTrue(os.path.exists(pair[1]))


class TestDeviceUtils(unittest.TestCase):
    """Test device utility functions."""
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("auto")
        self.assertIsNotNone(device)
        
        device = get_device("cpu")
        self.assertEqual(str(device), "cpu")
    
    def test_set_seed(self):
        """Test seed setting."""
        # This is hard to test directly, but we can ensure it doesn't raise errors
        set_seed(42)
        set_seed(123)
    
    def test_format_time(self):
        """Test time formatting."""
        from src.utils.device import format_time
        
        self.assertEqual(format_time(30), "30.0s")
        self.assertEqual(format_time(90), "1m 30.0s")
        self.assertEqual(format_time(3661), "1h 1m 1.0s")
    
    def test_ensure_dir(self):
        """Test directory creation."""
        from src.utils.device import ensure_dir
        
        temp_dir = os.path.join(self.temp_dir, "test_dir")
        ensure_dir(temp_dir)
        self.assertTrue(os.path.exists(temp_dir))
    
    def test_anonymize_filename(self):
        """Test filename anonymization."""
        from src.utils.device import anonymize_filename
        
        original = "my_song_file.wav"
        anonymized = anonymize_filename(original)
        
        self.assertNotEqual(original, anonymized)
        self.assertTrue(anonymized.endswith(".wav"))
        self.assertTrue(anonymized.startswith("audio_"))


if __name__ == "__main__":
    # Set up test environment
    unittest.main()
