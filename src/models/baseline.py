"""Baseline cover song identification model using traditional ML approaches."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.features.audio_features import AudioFeatureExtractor
from src.models.dtw_crp import CrossRecurrencePlot, DynamicTimeWarping
from src.utils.device import get_device


class BaselineCoverSongIdentifier:
    """Baseline cover song identification using traditional ML approaches.
    
    This class implements multiple approaches for cover song identification:
    1. Traditional ML classifiers (SVM, KNN, Random Forest)
    2. Dynamic Time Warping (DTW) for sequence alignment
    3. Cross-Recurrence Plot (CRP) for pattern analysis
    """
    
    def __init__(
        self,
        feature_extractor: Optional[AudioFeatureExtractor] = None,
        classifier_type: str = "svm",
        similarity_threshold: float = 0.5,
        device: Optional[str] = None
    ) -> None:
        """Initialize the baseline identifier.
        
        Args:
            feature_extractor: Audio feature extractor. If None, creates default.
            classifier_type: Type of classifier ('svm', 'knn', 'random_forest').
            similarity_threshold: Threshold for cover song classification.
            device: Device to use for computation.
        """
        self.feature_extractor = feature_extractor or AudioFeatureExtractor()
        self.classifier_type = classifier_type
        self.similarity_threshold = similarity_threshold
        self.device = get_device(device)
        
        # Initialize classifier
        self.classifier = self._create_classifier()
        
        # Initialize DTW and CRP
        self.dtw = DynamicTimeWarping(window_size=100)
        self.crp = CrossRecurrencePlot(threshold=0.1)
        
        # Model state
        self.is_trained = False
        self.feature_dim = self.feature_extractor.get_feature_dimension()
    
    def _create_classifier(self):
        """Create the specified classifier."""
        if self.classifier_type == "svm":
            return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
        elif self.classifier_type == "knn":
            return KNeighborsClassifier(n_neighbors=5, weights="distance")
        elif self.classifier_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, float]:
        """Train the cover song identifier.
        
        Args:
            X: Training features (n_samples, feature_dim).
            y: Training labels (n_samples,).
            validation_data: Optional validation data (X_val, y_val).
            
        Returns:
            Dictionary containing training metrics.
        """
        # Train classifier
        self.classifier.fit(X, y)
        self.is_trained = True
        
        # Compute training accuracy
        train_pred = self.classifier.predict(X)
        train_acc = np.mean(train_pred == y)
        
        metrics = {"train_accuracy": train_acc}
        
        # Compute validation metrics if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            val_pred = self.classifier.predict(X_val)
            val_acc = np.mean(val_pred == y_val)
            metrics["val_accuracy"] = val_acc
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cover song labels.
        
        Args:
            X: Features to predict (n_samples, feature_dim).
            
        Returns:
            Predicted labels.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict cover song probabilities.
        
        Args:
            X: Features to predict (n_samples, feature_dim).
            
        Returns:
            Predicted probabilities.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.classifier.predict_proba(X)
    
    def identify_cover_song(
        self,
        audio1_path: str,
        audio2_path: str,
        method: str = "combined"
    ) -> Dict[str, Union[float, bool, str]]:
        """Identify if two audio files are cover versions of each other.
        
        Args:
            audio1_path: Path to first audio file.
            audio2_path: Path to second audio file.
            method: Method to use ('classifier', 'dtw', 'crp', 'combined').
            
        Returns:
            Dictionary containing similarity score, prediction, and method used.
        """
        # Load and extract features
        features1 = self._extract_features_from_file(audio1_path)
        features2 = self._extract_features_from_file(audio2_path)
        
        if method == "classifier":
            return self._classifier_similarity(features1, features2)
        elif method == "dtw":
            return self._dtw_similarity(features1, features2)
        elif method == "crp":
            return self._crp_similarity(features1, features2)
        elif method == "combined":
            return self._combined_similarity(features1, features2)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _extract_features_from_file(self, audio_path: str) -> np.ndarray:
        """Extract features from audio file."""
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=self.feature_extractor.sample_rate)
            return self.feature_extractor.extract_combined_features(audio)
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return np.zeros(self.feature_dim)
    
    def _classifier_similarity(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> Dict[str, Union[float, bool, str]]:
        """Compute similarity using trained classifier."""
        if not self.is_trained:
            # Fallback to distance-based similarity
            distance = np.linalg.norm(features1 - features2)
            similarity = 1.0 / (1.0 + distance)
            is_cover = similarity > self.similarity_threshold
        else:
            # Use feature difference as input
            feature_diff = np.abs(features1 - features2)
            proba = self.classifier.predict_proba([feature_diff])[0]
            similarity = proba[1] if len(proba) > 1 else proba[0]
            is_cover = similarity > self.similarity_threshold
        
        return {
            "similarity": float(similarity),
            "is_cover": bool(is_cover),
            "method": "classifier"
        }
    
    def _dtw_similarity(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> Dict[str, Union[float, bool, str]]:
        """Compute similarity using Dynamic Time Warping."""
        # Reshape features for DTW (assuming they represent time series)
        # For static features, we'll treat them as single-frame sequences
        seq1 = features1.reshape(1, -1)
        seq2 = features2.reshape(1, -1)
        
        similarity = self.dtw.compute_similarity(seq1, seq2, normalize=True)
        is_cover = similarity > self.similarity_threshold
        
        return {
            "similarity": float(similarity),
            "is_cover": bool(is_cover),
            "method": "dtw"
        }
    
    def _crp_similarity(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> Dict[str, Union[float, bool, str]]:
        """Compute similarity using Cross-Recurrence Plot."""
        # Reshape features for CRP
        seq1 = features1.reshape(1, -1)
        seq2 = features2.reshape(1, -1)
        
        crp = self.crp.compute_crp(seq1, seq2)
        recurrence_rate = self.crp.compute_recurrence_rate(crp)
        determinism = self.crp.compute_determinism(crp)
        
        # Combine recurrence rate and determinism as similarity
        similarity = (recurrence_rate + determinism) / 2.0
        is_cover = similarity > self.similarity_threshold
        
        return {
            "similarity": float(similarity),
            "is_cover": bool(is_cover),
            "method": "crp"
        }
    
    def _combined_similarity(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> Dict[str, Union[float, bool, str]]:
        """Compute similarity using combined methods."""
        # Get similarities from different methods
        classifier_result = self._classifier_similarity(features1, features2)
        dtw_result = self._dtw_similarity(features1, features2)
        crp_result = self._crp_similarity(features1, features2)
        
        # Weighted combination
        weights = {"classifier": 0.5, "dtw": 0.3, "crp": 0.2}
        
        combined_similarity = (
            weights["classifier"] * classifier_result["similarity"] +
            weights["dtw"] * dtw_result["similarity"] +
            weights["crp"] * crp_result["similarity"]
        )
        
        is_cover = combined_similarity > self.similarity_threshold
        
        return {
            "similarity": float(combined_similarity),
            "is_cover": bool(is_cover),
            "method": "combined",
            "classifier_similarity": classifier_result["similarity"],
            "dtw_similarity": dtw_result["similarity"],
            "crp_similarity": crp_result["similarity"]
        }
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        if hasattr(self.classifier, "feature_importances_"):
            return self.classifier.feature_importances_
        elif hasattr(self.classifier, "coef_"):
            return np.abs(self.classifier.coef_[0])
        else:
            return None
    
    def save_model(self, path: str) -> None:
        """Save the trained model."""
        import pickle
        
        model_data = {
            "classifier": self.classifier,
            "feature_extractor": self.feature_extractor,
            "classifier_type": self.classifier_type,
            "similarity_threshold": self.similarity_threshold,
            "is_trained": self.is_trained,
            "feature_dim": self.feature_dim
        }
        
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path: str) -> None:
        """Load a trained model."""
        import pickle
        
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data["classifier"]
        self.feature_extractor = model_data["feature_extractor"]
        self.classifier_type = model_data["classifier_type"]
        self.similarity_threshold = model_data["similarity_threshold"]
        self.is_trained = model_data["is_trained"]
        self.feature_dim = model_data["feature_dim"]
