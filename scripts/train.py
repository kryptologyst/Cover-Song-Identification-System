#!/usr/bin/env python3
"""Training script for cover song identification models."""

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.synthetic_dataset import SyntheticCoverSongDataset
from src.features.audio_features import AudioFeatureExtractor
from src.metrics.evaluation import CoverSongMetrics
from src.models.baseline import BaselineCoverSongIdentifier
from src.utils.device import get_device, set_seed, ensure_dir


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config, resolve=True)


def prepare_data(config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare training and validation data."""
    # Load or generate dataset
    dataset = SyntheticCoverSongDataset(
        data_dir=config["data"]["data_dir"],
        num_songs=config["data"].get("num_songs", 100),
        num_covers_per_song=config["data"].get("num_covers_per_song", 3),
        privacy_mode=config.get("privacy_mode", True)
    )
    
    # Generate dataset if it doesn't exist
    if not os.path.exists(os.path.join(config["data"]["data_dir"], "meta.csv")):
        print("Generating synthetic dataset...")
        dataset.generate_dataset()
    
    # Load dataset metadata
    df = dataset.load_dataset()
    
    # Extract features
    feature_extractor = AudioFeatureExtractor(
        sample_rate=config["model"]["feature_extractor"]["sample_rate"],
        n_fft=config["model"]["feature_extractor"]["n_fft"],
        hop_length=config["model"]["feature_extractor"]["hop_length"],
        n_mels=config["model"]["feature_extractor"]["n_mels"],
        n_mfcc=config["model"]["feature_extractor"]["n_mfcc"],
        chroma_bins=config["model"]["feature_extractor"]["chroma_bins"],
        spectral_contrast_bins=config["model"]["feature_extractor"]["spectral_contrast_bins"]
    )
    
    print("Extracting features...")
    features = []
    labels = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            import librosa
            audio, sr = librosa.load(row["path"], sr=feature_extractor.sample_rate)
            feature_vector = feature_extractor.extract_combined_features(audio)
            features.append(feature_vector)
            
            # Create labels: 1 for cover, 0 for original
            label = 1 if row["label"] == "cover" else 0
            labels.append(label)
        except Exception as e:
            print(f"Error processing {row['path']}: {e}")
            continue
    
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Cover songs: {np.sum(labels)}")
    print(f"Original songs: {np.sum(1 - labels)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=config["data"]["test_split"],
        random_state=config["seed"],
        stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=config["data"]["val_split"] / (1 - config["data"]["test_split"]),
        random_state=config["seed"],
        stratify=y_train
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict
) -> BaselineCoverSongIdentifier:
    """Train the cover song identification model."""
    # Initialize model
    feature_extractor = AudioFeatureExtractor(
        sample_rate=config["model"]["feature_extractor"]["sample_rate"],
        n_fft=config["model"]["feature_extractor"]["n_fft"],
        hop_length=config["model"]["feature_extractor"]["hop_length"],
        n_mels=config["model"]["feature_extractor"]["n_mels"],
        n_mfcc=config["model"]["feature_extractor"]["n_mfcc"],
        chroma_bins=config["model"]["feature_extractor"]["chroma_bins"],
        spectral_contrast_bins=config["model"]["feature_extractor"]["spectral_contrast_bins"]
    )
    
    model = BaselineCoverSongIdentifier(
        feature_extractor=feature_extractor,
        classifier_type=config["model"]["classifier"]["_target_"].split(".")[-1].lower(),
        similarity_threshold=config["model"]["similarity_threshold"],
        device=get_device(config.get("device", "auto"))
    )
    
    print(f"Training {config['model']['classifier']['_target_'].split('.')[-1]} model...")
    
    # Train model
    train_metrics = model.train(
        X_train, y_train,
        validation_data=(X_val, y_val)
    )
    
    print("Training completed!")
    print(f"Training accuracy: {train_metrics['train_accuracy']:.4f}")
    if "val_accuracy" in train_metrics:
        print(f"Validation accuracy: {train_metrics['val_accuracy']:.4f}")
    
    return model


def evaluate_model(
    model: BaselineCoverSongIdentifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict
) -> Dict:
    """Evaluate the trained model."""
    print("Evaluating model...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Compute metrics
    metrics_calculator = CoverSongMetrics(k_values=config["evaluation"]["k_values"])
    
    # For ranking metrics, we need similarity matrix
    # Create a simple similarity matrix based on predictions
    n_test = len(X_test)
    similarities = np.zeros((n_test, n_test))
    labels_matrix = np.zeros((n_test, n_test))
    
    for i in range(n_test):
        for j in range(n_test):
            if i != j:
                # Use probability as similarity
                similarities[i, j] = y_proba[i][1] if len(y_proba[i]) > 1 else y_proba[i][0]
                # Labels: 1 if both are same type (both cover or both original)
                labels_matrix[i, j] = 1 if y_test[i] == y_test[j] else 0
    
    # Compute all metrics
    all_metrics = metrics_calculator.compute_all_metrics(similarities, labels_matrix)
    
    # Compute confusion matrix metrics
    confusion_metrics = metrics_calculator.compute_confusion_matrix(
        similarities.flatten(),
        labels_matrix.flatten(),
        threshold=config["evaluation"]["similarity_threshold"]
    )
    
    # Combine metrics
    evaluation_results = {
        **all_metrics,
        **confusion_metrics
    }
    
    # Print evaluation report
    report = metrics_calculator.generate_evaluation_report(
        similarities, labels_matrix,
        threshold=config["evaluation"]["similarity_threshold"]
    )
    print(report)
    
    return evaluation_results


def save_results(
    model: BaselineCoverSongIdentifier,
    evaluation_results: Dict,
    config: Dict,
    output_dir: str
) -> None:
    """Save model and evaluation results."""
    ensure_dir(output_dir)
    
    # Save model
    model_path = os.path.join(output_dir, "best_model.pkl")
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save evaluation results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    import json
    with open(results_path, "w") as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    print(f"Evaluation results saved to: {results_path}")
    
    # Save configuration
    config_path = os.path.join(output_dir, "config.yaml")
    OmegaConf.save(config, config_path)
    print(f"Configuration saved to: {config_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train cover song identification model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Output directory for model and results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    config["seed"] = args.seed
    
    print(f"Configuration loaded from: {args.config}")
    print(f"Output directory: {args.output_dir}")
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(config)
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val, config)
    
    # Evaluate model
    evaluation_results = evaluate_model(model, X_test, y_test, config)
    
    # Save results
    save_results(model, evaluation_results, config, args.output_dir)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
