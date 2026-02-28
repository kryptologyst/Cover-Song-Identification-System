"""Comprehensive evaluation metrics for cover song identification."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, confusion_matrix


class CoverSongMetrics:
    """Comprehensive evaluation metrics for cover song identification.
    
    This class implements various metrics commonly used in cover song
    identification research, including mAP@k, R@k, DTW distance statistics,
    and ROC analysis.
    """
    
    def __init__(self, k_values: List[int] = [1, 5, 10, 20]) -> None:
        """Initialize metrics calculator.
        
        Args:
            k_values: List of k values for mAP@k and R@k calculations.
        """
        self.k_values = k_values
    
    def compute_all_metrics(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        query_ids: Optional[np.ndarray] = None
    ) -> Dict[str, Union[float, Dict[int, float]]]:
        """Compute all evaluation metrics.
        
        Args:
            similarities: Similarity scores (n_queries, n_database).
            labels: Binary labels indicating cover relationships (n_queries, n_database).
            query_ids: Query identifiers for per-query analysis.
            
        Returns:
            Dictionary containing all computed metrics.
        """
        metrics = {}
        
        # Basic classification metrics
        metrics.update(self._compute_classification_metrics(similarities, labels))
        
        # Ranking metrics
        metrics.update(self._compute_ranking_metrics(similarities, labels))
        
        # Per-query metrics
        if query_ids is not None:
            metrics.update(self._compute_per_query_metrics(similarities, labels, query_ids))
        
        # DTW distance statistics
        metrics.update(self._compute_dtw_statistics(similarities, labels))
        
        return metrics
    
    def _compute_classification_metrics(
        self,
        similarities: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute basic classification metrics."""
        metrics = {}
        
        # Flatten arrays for binary classification
        similarities_flat = similarities.flatten()
        labels_flat = labels.flatten()
        
        # ROC AUC
        if len(np.unique(labels_flat)) > 1:
            metrics["roc_auc"] = roc_auc_score(labels_flat, similarities_flat)
            
            # Find optimal threshold
            fpr, tpr, thresholds = roc_curve(labels_flat, similarities_flat)
            optimal_idx = np.argmax(tpr - fpr)
            metrics["optimal_threshold"] = thresholds[optimal_idx]
            metrics["optimal_tpr"] = tpr[optimal_idx]
            metrics["optimal_fpr"] = fpr[optimal_idx]
        
        # Average Precision (mAP)
        metrics["map"] = average_precision_score(labels_flat, similarities_flat)
        
        # Precision-Recall curve metrics
        precision, recall, pr_thresholds = precision_recall_curve(labels_flat, similarities_flat)
        
        # F1 score at optimal threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_pr_idx = np.argmax(f1_scores)
        metrics["optimal_f1"] = f1_scores[optimal_pr_idx]
        metrics["optimal_precision"] = precision[optimal_pr_idx]
        metrics["optimal_recall"] = recall[optimal_pr_idx]
        
        return metrics
    
    def _compute_ranking_metrics(
        self,
        similarities: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Dict[int, float]]:
        """Compute ranking-based metrics (mAP@k, R@k)."""
        metrics = {}
        
        n_queries, n_database = similarities.shape
        
        # Initialize metric dictionaries
        map_at_k = {}
        recall_at_k = {}
        
        for k in self.k_values:
            map_scores = []
            recall_scores = []
            
            for query_idx in range(n_queries):
                # Get similarities and labels for this query
                query_similarities = similarities[query_idx]
                query_labels = labels[query_idx]
                
                # Sort by similarity (descending)
                sorted_indices = np.argsort(query_similarities)[::-1]
                sorted_labels = query_labels[sorted_indices]
                
                # Compute mAP@k
                map_k = self._compute_map_at_k(sorted_labels, k)
                map_scores.append(map_k)
                
                # Compute R@k
                recall_k = self._compute_recall_at_k(sorted_labels, k)
                recall_scores.append(recall_k)
            
            map_at_k[k] = np.mean(map_scores)
            recall_at_k[k] = np.mean(recall_scores)
        
        metrics["map_at_k"] = map_at_k
        metrics["recall_at_k"] = recall_at_k
        
        return metrics
    
    def _compute_map_at_k(self, sorted_labels: np.ndarray, k: int) -> float:
        """Compute Mean Average Precision at k."""
        if k > len(sorted_labels):
            k = len(sorted_labels)
        
        # Count relevant items in top-k
        relevant_items = np.sum(sorted_labels[:k])
        
        if relevant_items == 0:
            return 0.0
        
        # Compute precision at each position
        precisions = []
        for i in range(k):
            if sorted_labels[i] == 1:
                # Precision at position i+1
                precision = np.sum(sorted_labels[:i+1]) / (i + 1)
                precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def _compute_recall_at_k(self, sorted_labels: np.ndarray, k: int) -> float:
        """Compute Recall at k."""
        total_relevant = np.sum(sorted_labels)
        
        if total_relevant == 0:
            return 0.0
        
        if k > len(sorted_labels):
            k = len(sorted_labels)
        
        relevant_in_top_k = np.sum(sorted_labels[:k])
        return relevant_in_top_k / total_relevant
    
    def _compute_per_query_metrics(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        query_ids: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-query metrics."""
        metrics = {}
        
        unique_queries = np.unique(query_ids)
        per_query_map = {}
        per_query_recall = {}
        
        for query_id in unique_queries:
            query_mask = query_ids == query_id
            query_similarities = similarities[query_mask]
            query_labels = labels[query_mask]
            
            # Compute metrics for this query
            query_map_scores = []
            query_recall_scores = []
            
            for k in self.k_values:
                # Average across all instances of this query
                map_scores = []
                recall_scores = []
                
                for i in range(len(query_similarities)):
                    sorted_indices = np.argsort(query_similarities[i])[::-1]
                    sorted_labels = query_labels[i][sorted_indices]
                    
                    map_k = self._compute_map_at_k(sorted_labels, k)
                    recall_k = self._compute_recall_at_k(sorted_labels, k)
                    
                    map_scores.append(map_k)
                    recall_scores.append(recall_k)
                
                query_map_scores.append(np.mean(map_scores))
                query_recall_scores.append(np.mean(recall_scores))
            
            per_query_map[str(query_id)] = {
                f"map_at_{k}": score for k, score in zip(self.k_values, query_map_scores)
            }
            per_query_recall[str(query_id)] = {
                f"recall_at_{k}": score for k, score in zip(self.k_values, query_recall_scores)
            }
        
        metrics["per_query_map"] = per_query_map
        metrics["per_query_recall"] = per_query_recall
        
        return metrics
    
    def _compute_dtw_statistics(
        self,
        similarities: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute DTW distance statistics."""
        metrics = {}
        
        # Separate similarities for positive and negative pairs
        positive_similarities = similarities[labels == 1]
        negative_similarities = similarities[labels == 0]
        
        if len(positive_similarities) > 0:
            metrics["positive_mean_similarity"] = np.mean(positive_similarities)
            metrics["positive_std_similarity"] = np.std(positive_similarities)
            metrics["positive_median_similarity"] = np.median(positive_similarities)
            metrics["positive_min_similarity"] = np.min(positive_similarities)
            metrics["positive_max_similarity"] = np.max(positive_similarities)
        
        if len(negative_similarities) > 0:
            metrics["negative_mean_similarity"] = np.mean(negative_similarities)
            metrics["negative_std_similarity"] = np.std(negative_similarities)
            metrics["negative_median_similarity"] = np.median(negative_similarities)
            metrics["negative_min_similarity"] = np.min(negative_similarities)
            metrics["negative_max_similarity"] = np.max(negative_similarities)
        
        # Separation between positive and negative distributions
        if len(positive_similarities) > 0 and len(negative_similarities) > 0:
            separation = np.mean(positive_similarities) - np.mean(negative_similarities)
            metrics["distribution_separation"] = separation
            
            # Overlap measure
            overlap = self._compute_distribution_overlap(
                positive_similarities, negative_similarities
            )
            metrics["distribution_overlap"] = overlap
        
        return metrics
    
    def _compute_distribution_overlap(
        self,
        dist1: np.ndarray,
        dist2: np.ndarray
    ) -> float:
        """Compute overlap between two distributions."""
        # Use histogram-based approach
        min_val = min(np.min(dist1), np.min(dist2))
        max_val = max(np.max(dist1), np.max(dist2))
        
        bins = np.linspace(min_val, max_val, 100)
        
        hist1, _ = np.histogram(dist1, bins=bins, density=True)
        hist2, _ = np.histogram(dist2, bins=bins, density=True)
        
        # Compute overlap as minimum of the two histograms
        overlap = np.sum(np.minimum(hist1, hist2)) * (bins[1] - bins[0])
        
        return overlap
    
    def compute_confusion_matrix(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Union[int, float]]:
        """Compute confusion matrix and derived metrics."""
        predictions = (similarities >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(labels.flatten(), predictions.flatten()).ravel()
        
        metrics = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "f1_score": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        }
        
        return metrics
    
    def generate_evaluation_report(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        query_ids: Optional[np.ndarray] = None,
        threshold: float = 0.5
    ) -> str:
        """Generate a comprehensive evaluation report."""
        all_metrics = self.compute_all_metrics(similarities, labels, query_ids)
        confusion_metrics = self.compute_confusion_matrix(similarities, labels, threshold)
        
        report = []
        report.append("=" * 60)
        report.append("COVER SONG IDENTIFICATION EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Classification metrics
        report.append("CLASSIFICATION METRICS:")
        report.append(f"  ROC AUC: {all_metrics.get('roc_auc', 'N/A'):.4f}")
        report.append(f"  mAP: {all_metrics.get('map', 'N/A'):.4f}")
        report.append(f"  Optimal F1: {all_metrics.get('optimal_f1', 'N/A'):.4f}")
        report.append(f"  Optimal Precision: {all_metrics.get('optimal_precision', 'N/A'):.4f}")
        report.append(f"  Optimal Recall: {all_metrics.get('optimal_recall', 'N/A'):.4f}")
        report.append("")
        
        # Ranking metrics
        report.append("RANKING METRICS:")
        map_at_k = all_metrics.get("map_at_k", {})
        recall_at_k = all_metrics.get("recall_at_k", {})
        
        for k in self.k_values:
            report.append(f"  mAP@{k}: {map_at_k.get(k, 'N/A'):.4f}")
            report.append(f"  R@{k}: {recall_at_k.get(k, 'N/A'):.4f}")
        report.append("")
        
        # Confusion matrix metrics
        report.append("CONFUSION MATRIX METRICS (threshold={:.3f}):".format(threshold))
        report.append(f"  Accuracy: {confusion_metrics['accuracy']:.4f}")
        report.append(f"  Precision: {confusion_metrics['precision']:.4f}")
        report.append(f"  Recall: {confusion_metrics['recall']:.4f}")
        report.append(f"  Specificity: {confusion_metrics['specificity']:.4f}")
        report.append(f"  F1 Score: {confusion_metrics['f1_score']:.4f}")
        report.append("")
        
        # DTW statistics
        report.append("SIMILARITY DISTRIBUTION STATISTICS:")
        if "positive_mean_similarity" in all_metrics:
            report.append(f"  Positive pairs - Mean: {all_metrics['positive_mean_similarity']:.4f}")
            report.append(f"  Positive pairs - Std: {all_metrics['positive_std_similarity']:.4f}")
            report.append(f"  Positive pairs - Median: {all_metrics['positive_median_similarity']:.4f}")
        
        if "negative_mean_similarity" in all_metrics:
            report.append(f"  Negative pairs - Mean: {all_metrics['negative_mean_similarity']:.4f}")
            report.append(f"  Negative pairs - Std: {all_metrics['negative_std_similarity']:.4f}")
            report.append(f"  Negative pairs - Median: {all_metrics['negative_median_similarity']:.4f}")
        
        if "distribution_separation" in all_metrics:
            report.append(f"  Distribution Separation: {all_metrics['distribution_separation']:.4f}")
            report.append(f"  Distribution Overlap: {all_metrics['distribution_overlap']:.4f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
