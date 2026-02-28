#!/usr/bin/env python3
"""Evaluation script for cover song identification models."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.data.synthetic_dataset import SyntheticCoverSongDataset
from src.features.audio_features import AudioFeatureExtractor
from src.metrics.evaluation import CoverSongMetrics
from src.models.baseline import BaselineCoverSongIdentifier
from src.utils.device import get_device, set_seed


def load_model(model_path: str) -> BaselineCoverSongIdentifier:
    """Load a trained model."""
    model = BaselineCoverSongIdentifier()
    model.load_model(model_path)
    return model


def load_test_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load test data and extract features."""
    # Load dataset
    dataset = SyntheticCoverSongDataset(data_dir=data_dir)
    df = dataset.load_dataset()
    
    # Filter test split
    test_df = df[df["split"] == "test"]
    
    if len(test_df) == 0:
        raise ValueError("No test data found. Make sure dataset has test split.")
    
    print(f"Loading {len(test_df)} test files...")
    
    # Extract features
    feature_extractor = AudioFeatureExtractor()
    features = []
    labels = []
    file_paths = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        try:
            audio, sr = librosa.load(row["path"], sr=feature_extractor.sample_rate)
            feature_vector = feature_extractor.extract_combined_features(audio)
            features.append(feature_vector)
            
            # Create labels: 1 for cover, 0 for original
            label = 1 if row["label"] == "cover" else 0
            labels.append(label)
            file_paths.append(row["path"])
        except Exception as e:
            print(f"Error processing {row['path']}: {e}")
            continue
    
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"Loaded {len(features)} test samples")
    print(f"Cover songs: {np.sum(labels)}")
    print(f"Original songs: {np.sum(1 - labels)}")
    
    return features, labels, file_paths


def evaluate_on_pairs(
    model: BaselineCoverSongIdentifier,
    features: np.ndarray,
    labels: np.ndarray,
    file_paths: List[str],
    config: Dict
) -> Dict:
    """Evaluate model on cover song pairs."""
    print("Evaluating on cover song pairs...")
    
    # Create pairs for evaluation
    pairs = []
    pair_labels = []
    
    # Group by song_id to find original-cover pairs
    song_groups = {}
    for i, path in enumerate(file_paths):
        # Extract song_id from filename (assuming format: song_XXX_version.wav)
        filename = os.path.basename(path)
        if "song_" in filename:
            song_id = filename.split("_")[1]
            if song_id not in song_groups:
                song_groups[song_id] = {"original": [], "covers": []}
            
            if "original" in filename:
                song_groups[song_id]["original"].append(i)
            elif "cover" in filename:
                song_groups[song_id]["covers"].append(i)
    
    # Create positive pairs (original-cover)
    for song_id, group in song_groups.items():
        if group["original"] and group["covers"]:
            for orig_idx in group["original"]:
                for cover_idx in group["covers"]:
                    pairs.append((orig_idx, cover_idx))
                    pair_labels.append(1)  # Positive pair
    
    # Create negative pairs (different songs)
    song_ids = list(song_groups.keys())
    for i, song_id1 in enumerate(song_ids):
        for song_id2 in song_ids[i+1:]:
            if song_groups[song_id1]["original"] and song_groups[song_id2]["original"]:
                orig1_idx = song_groups[song_id1]["original"][0]
                orig2_idx = song_groups[song_id2]["original"][0]
                pairs.append((orig1_idx, orig2_idx))
                pair_labels.append(0)  # Negative pair
    
    print(f"Created {len(pairs)} pairs for evaluation")
    print(f"Positive pairs: {np.sum(pair_labels)}")
    print(f"Negative pairs: {np.sum(1 - np.array(pair_labels))}")
    
    # Evaluate pairs
    similarities = []
    predictions = []
    
    for orig_idx, cover_idx in tqdm(pairs):
        # Get features
        feat1 = features[orig_idx]
        feat2 = features[cover_idx]
        
        # Compute similarity using different methods
        if config["method"] == "classifier":
            # Use classifier-based similarity
            feature_diff = np.abs(feat1 - feat2)
            proba = model.classifier.predict_proba([feature_diff])[0]
            similarity = proba[1] if len(proba) > 1 else proba[0]
        elif config["method"] == "dtw":
            # Use DTW similarity
            seq1 = feat1.reshape(1, -1)
            seq2 = feat2.reshape(1, -1)
            similarity = model.dtw.compute_similarity(seq1, seq2, normalize=True)
        elif config["method"] == "crp":
            # Use CRP similarity
            seq1 = feat1.reshape(1, -1)
            seq2 = feat2.reshape(1, -1)
            crp = model.crp.compute_crp(seq1, seq2)
            recurrence_rate = model.crp.compute_recurrence_rate(crp)
            determinism = model.crp.compute_determinism(crp)
            similarity = (recurrence_rate + determinism) / 2.0
        elif config["method"] == "combined":
            # Use combined similarity
            result = model._combined_similarity(feat1, feat2)
            similarity = result["similarity"]
        else:
            # Default to cosine similarity
            similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
        
        similarities.append(similarity)
        predictions.append(1 if similarity > config["threshold"] else 0)
    
    similarities = np.array(similarities)
    predictions = np.array(predictions)
    pair_labels = np.array(pair_labels)
    
    # Compute metrics
    metrics_calculator = CoverSongMetrics(k_values=config["k_values"])
    
    # Create similarity matrix for ranking metrics
    n_pairs = len(pairs)
    similarity_matrix = np.zeros((n_pairs, n_pairs))
    label_matrix = np.zeros((n_pairs, n_pairs))
    
    for i in range(n_pairs):
        for j in range(n_pairs):
            if i != j:
                similarity_matrix[i, j] = similarities[i]
                label_matrix[i, j] = 1 if pair_labels[i] == pair_labels[j] else 0
    
    # Compute all metrics
    all_metrics = metrics_calculator.compute_all_metrics(
        similarity_matrix, label_matrix
    )
    
    # Compute confusion matrix metrics
    confusion_metrics = metrics_calculator.compute_confusion_matrix(
        similarities, pair_labels, threshold=config["threshold"]
    )
    
    # Combine metrics
    evaluation_results = {
        **all_metrics,
        **confusion_metrics,
        "method": config["method"],
        "threshold": config["threshold"],
        "num_pairs": len(pairs),
        "num_positive_pairs": np.sum(pair_labels),
        "num_negative_pairs": np.sum(1 - pair_labels)
    }
    
    return evaluation_results


def run_ablation_study(
    model: BaselineCoverSongIdentifier,
    features: np.ndarray,
    labels: np.ndarray,
    file_paths: List[str],
    config: Dict
) -> Dict:
    """Run ablation study with different methods and thresholds."""
    print("Running ablation study...")
    
    methods = ["classifier", "dtw", "crp", "combined"]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    ablation_results = {}
    
    for method in methods:
        print(f"Evaluating method: {method}")
        method_results = {}
        
        for threshold in thresholds:
            config_copy = config.copy()
            config_copy["method"] = method
            config_copy["threshold"] = threshold
            
            results = evaluate_on_pairs(model, features, labels, file_paths, config_copy)
            method_results[f"threshold_{threshold}"] = results
        
        ablation_results[method] = method_results
    
    return ablation_results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate cover song identification model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data",
        help="Path to test data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="combined",
        choices=["classifier", "dtw", "crp", "combined"],
        help="Evaluation method"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Similarity threshold"
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study"
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="K values for ranking metrics"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(42)
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path)
    
    # Load test data
    features, labels, file_paths = load_test_data(args.test_data)
    
    # Prepare configuration
    config = {
        "method": args.method,
        "threshold": args.threshold,
        "k_values": args.k_values
    }
    
    # Run evaluation
    if args.ablation:
        results = run_ablation_study(model, features, labels, file_paths, config)
    else:
        results = evaluate_on_pairs(model, features, labels, file_paths, config)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.ablation:
        output_path = os.path.join(args.output_dir, "ablation_results.json")
    else:
        output_path = os.path.join(args.output_dir, "evaluation_results.json")
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {output_path}")
    
    # Print summary
    if not args.ablation:
        print("\nEvaluation Summary:")
        print(f"Method: {args.method}")
        print(f"Threshold: {args.threshold}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"ROC AUC: {results.get('roc_auc', 'N/A')}")
        print(f"mAP: {results.get('map', 'N/A')}")
    
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
