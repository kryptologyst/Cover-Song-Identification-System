"""Audio feature extraction utilities."""

from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torchaudio
from scipy import signal


class AudioFeatureExtractor:
    """Extract comprehensive audio features for cover song identification.
    
    This class extracts multiple types of audio features including MFCC, chroma,
    spectral contrast, tempo, and rhythm patterns that are useful for identifying
    cover songs.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        n_mfcc: int = 13,
        chroma_bins: int = 12,
        spectral_contrast_bins: int = 7,
        tempo_bins: int = 1,
        rhythm_bins: int = 1,
    ) -> None:
        """Initialize the feature extractor.
        
        Args:
            sample_rate: Target sample rate for audio processing.
            n_fft: FFT window size.
            hop_length: Number of samples between successive frames.
            n_mels: Number of mel bands.
            n_mfcc: Number of MFCC coefficients.
            chroma_bins: Number of chroma bins.
            spectral_contrast_bins: Number of spectral contrast bins.
            tempo_bins: Number of tempo bins.
            rhythm_bins: Number of rhythm bins.
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.chroma_bins = chroma_bins
        self.spectral_contrast_bins = spectral_contrast_bins
        self.tempo_bins = tempo_bins
        self.rhythm_bins = rhythm_bins
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all audio features from audio signal.
        
        Args:
            audio: Audio signal as numpy array.
            
        Returns:
            Dict containing all extracted features.
        """
        features = {}
        
        # Ensure audio is at correct sample rate
        if len(audio) == 0:
            # Return zero features for empty audio
            return self._get_zero_features()
        
        # Extract MFCC features
        features["mfcc"] = self._extract_mfcc(audio)
        
        # Extract chroma features
        features["chroma"] = self._extract_chroma(audio)
        
        # Extract spectral contrast
        features["spectral_contrast"] = self._extract_spectral_contrast(audio)
        
        # Extract tempo
        features["tempo"] = self._extract_tempo(audio)
        
        # Extract rhythm patterns
        features["rhythm"] = self._extract_rhythm(audio)
        
        # Extract zero crossing rate
        features["zcr"] = self._extract_zcr(audio)
        
        # Extract spectral centroid
        features["spectral_centroid"] = self._extract_spectral_centroid(audio)
        
        # Extract spectral rolloff
        features["spectral_rolloff"] = self._extract_spectral_rolloff(audio)
        
        return features
    
    def _extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features."""
        try:
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
            # Return mean and std across time
            return np.concatenate([
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1)
            ])
        except Exception:
            return np.zeros(self.n_mfcc * 2)
    
    def _extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """Extract chroma features."""
        try:
            chroma = librosa.feature.chroma_stft(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_chroma=self.chroma_bins,
            )
            # Return mean and std across time
            return np.concatenate([
                np.mean(chroma, axis=1),
                np.std(chroma, axis=1)
            ])
        except Exception:
            return np.zeros(self.chroma_bins * 2)
    
    def _extract_spectral_contrast(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral contrast features."""
        try:
            contrast = librosa.feature.spectral_contrast(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_bands=self.spectral_contrast_bins,
            )
            # Return mean and std across time
            return np.concatenate([
                np.mean(contrast, axis=1),
                np.std(contrast, axis=1)
            ])
        except Exception:
            return np.zeros(self.spectral_contrast_bins * 2)
    
    def _extract_tempo(self, audio: np.ndarray) -> np.ndarray:
        """Extract tempo features."""
        try:
            tempo, _ = librosa.beat.beat_track(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
            )
            return np.array([tempo])
        except Exception:
            return np.zeros(1)
    
    def _extract_rhythm(self, audio: np.ndarray) -> np.ndarray:
        """Extract rhythm pattern features."""
        try:
            # Extract onset strength
            onset_strength = librosa.onset.onset_strength(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
            )
            # Return statistical features
            return np.array([
                np.mean(onset_strength),
                np.std(onset_strength),
                np.max(onset_strength),
                np.min(onset_strength),
            ])
        except Exception:
            return np.zeros(4)
    
    def _extract_zcr(self, audio: np.ndarray) -> np.ndarray:
        """Extract zero crossing rate."""
        try:
            zcr = librosa.feature.zero_crossing_rate(
                audio,
                hop_length=self.hop_length,
            )
            return np.array([np.mean(zcr), np.std(zcr)])
        except Exception:
            return np.zeros(2)
    
    def _extract_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral centroid."""
        try:
            centroid = librosa.feature.spectral_centroid(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
            )
            return np.array([np.mean(centroid), np.std(centroid)])
        except Exception:
            return np.zeros(2)
    
    def _extract_spectral_rolloff(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral rolloff."""
        try:
            rolloff = librosa.feature.spectral_rolloff(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
            )
            return np.array([np.mean(rolloff), np.std(rolloff)])
        except Exception:
            return np.zeros(2)
    
    def _get_zero_features(self) -> Dict[str, np.ndarray]:
        """Return zero features for empty audio."""
        return {
            "mfcc": np.zeros(self.n_mfcc * 2),
            "chroma": np.zeros(self.chroma_bins * 2),
            "spectral_contrast": np.zeros(self.spectral_contrast_bins * 2),
            "tempo": np.zeros(1),
            "rhythm": np.zeros(4),
            "zcr": np.zeros(2),
            "spectral_centroid": np.zeros(2),
            "spectral_rolloff": np.zeros(2),
        }
    
    def extract_combined_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract and combine all features into a single vector.
        
        Args:
            audio: Audio signal as numpy array.
            
        Returns:
            Combined feature vector.
        """
        features = self.extract_features(audio)
        return np.concatenate(list(features.values()))
    
    def get_feature_dimension(self) -> int:
        """Get the total dimension of combined features.
        
        Returns:
            Total feature dimension.
        """
        return (
            self.n_mfcc * 2 +
            self.chroma_bins * 2 +
            self.spectral_contrast_bins * 2 +
            1 +  # tempo
            4 +  # rhythm
            2 +  # zcr
            2 +  # spectral_centroid
            2    # spectral_rolloff
        )
