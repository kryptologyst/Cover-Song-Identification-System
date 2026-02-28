"""Dynamic Time Warping (DTW) implementation for cover song identification."""

from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import euclidean


class DynamicTimeWarping:
    """Dynamic Time Warping for sequence alignment and similarity computation.
    
    DTW is particularly useful for cover song identification as it can handle
    temporal variations in tempo and rhythm between original and cover versions.
    """
    
    def __init__(self, window_size: Optional[int] = None) -> None:
        """Initialize DTW with optional window constraint.
        
        Args:
            window_size: Maximum distance from diagonal in DTW matrix.
                        If None, no window constraint is applied.
        """
        self.window_size = window_size
    
    def compute_distance(
        self,
        seq1: np.ndarray,
        seq2: np.ndarray,
        distance_func: str = "euclidean"
    ) -> Tuple[float, np.ndarray]:
        """Compute DTW distance between two sequences.
        
        Args:
            seq1: First sequence (N x D feature matrix).
            seq2: Second sequence (M x D feature matrix).
            distance_func: Distance function to use ('euclidean', 'cosine').
            
        Returns:
            Tuple of (DTW distance, alignment path).
        """
        n, m = len(seq1), len(seq2)
        
        # Initialize distance matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Apply window constraint if specified
        if self.window_size is not None:
            for i in range(1, n + 1):
                start = max(1, i - self.window_size)
                end = min(m + 1, i + self.window_size + 1)
                for j in range(start, end):
                    cost = self._compute_cost(seq1[i-1], seq2[j-1], distance_func)
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],      # insertion
                        dtw_matrix[i, j-1],      # deletion
                        dtw_matrix[i-1, j-1]     # match
                    )
        else:
            # Standard DTW without window constraint
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = self._compute_cost(seq1[i-1], seq2[j-1], distance_func)
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],      # insertion
                        dtw_matrix[i, j-1],      # deletion
                        dtw_matrix[i-1, j-1]     # match
                    )
        
        # Trace back to find alignment path
        path = self._trace_back(dtw_matrix)
        
        return dtw_matrix[n, m], path
    
    def _compute_cost(
        self,
        point1: np.ndarray,
        point2: np.ndarray,
        distance_func: str
    ) -> float:
        """Compute cost between two points.
        
        Args:
            point1: First point.
            point2: Second point.
            distance_func: Distance function to use.
            
        Returns:
            Cost between the two points.
        """
        if distance_func == "euclidean":
            return euclidean(point1, point2)
        elif distance_func == "cosine":
            # Cosine distance
            dot_product = np.dot(point1, point2)
            norm1 = np.linalg.norm(point1)
            norm2 = np.linalg.norm(point2)
            if norm1 == 0 or norm2 == 0:
                return 1.0
            cosine_sim = dot_product / (norm1 * norm2)
            return 1 - cosine_sim
        else:
            raise ValueError(f"Unknown distance function: {distance_func}")
    
    def _trace_back(self, dtw_matrix: np.ndarray) -> np.ndarray:
        """Trace back the optimal alignment path.
        
        Args:
            dtw_matrix: DTW distance matrix.
            
        Returns:
            Alignment path as array of (i, j) coordinates.
        """
        n, m = dtw_matrix.shape
        path = []
        i, j = n - 1, m - 1
        
        while i > 0 or j > 0:
            path.append((i - 1, j - 1))  # Convert to 0-indexed
            
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                # Find the minimum cost direction
                min_cost = min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion
                    dtw_matrix[i-1, j-1]     # match
                )
                
                if dtw_matrix[i-1, j-1] == min_cost:
                    i -= 1
                    j -= 1
                elif dtw_matrix[i-1, j] == min_cost:
                    i -= 1
                else:
                    j -= 1
        
        path.reverse()
        return np.array(path)
    
    def compute_similarity(
        self,
        seq1: np.ndarray,
        seq2: np.ndarray,
        normalize: bool = True
    ) -> float:
        """Compute normalized DTW similarity score.
        
        Args:
            seq1: First sequence.
            seq2: Second sequence.
            normalize: Whether to normalize the distance.
            
        Returns:
            Similarity score between 0 and 1 (higher = more similar).
        """
        distance, _ = self.compute_distance(seq1, seq2)
        
        if normalize:
            # Normalize by sequence lengths
            n, m = len(seq1), len(seq2)
            normalized_distance = distance / (n + m)
            # Convert to similarity (0 = identical, 1 = very different)
            similarity = 1.0 / (1.0 + normalized_distance)
            return similarity
        else:
            return distance


class CrossRecurrencePlot:
    """Cross-Recurrence Plot (CRP) for cover song identification.
    
    CRP captures the recurrence patterns between two time series,
    which can be useful for identifying cover songs with different
    temporal structures.
    """
    
    def __init__(self, threshold: float = 0.1, metric: str = "euclidean") -> None:
        """Initialize CRP with threshold and distance metric.
        
        Args:
            threshold: Threshold for recurrence detection.
            metric: Distance metric to use.
        """
        self.threshold = threshold
        self.metric = metric
    
    def compute_crp(
        self,
        seq1: np.ndarray,
        seq2: np.ndarray
    ) -> np.ndarray:
        """Compute cross-recurrence plot.
        
        Args:
            seq1: First sequence.
            seq2: Second sequence.
            
        Returns:
            Binary recurrence matrix.
        """
        n, m = len(seq1), len(seq2)
        crp = np.zeros((n, m), dtype=bool)
        
        for i in range(n):
            for j in range(m):
                if self.metric == "euclidean":
                    distance = euclidean(seq1[i], seq2[j])
                elif self.metric == "cosine":
                    dot_product = np.dot(seq1[i], seq2[j])
                    norm1 = np.linalg.norm(seq1[i])
                    norm2 = np.linalg.norm(seq2[j])
                    if norm1 == 0 or norm2 == 0:
                        distance = 1.0
                    else:
                        cosine_sim = dot_product / (norm1 * norm2)
                        distance = 1 - cosine_sim
                else:
                    raise ValueError(f"Unknown metric: {self.metric}")
                
                crp[i, j] = distance <= self.threshold
        
        return crp
    
    def compute_recurrence_rate(self, crp: np.ndarray) -> float:
        """Compute recurrence rate from CRP.
        
        Args:
            crp: Cross-recurrence plot matrix.
            
        Returns:
            Recurrence rate (fraction of recurrent points).
        """
        return np.sum(crp) / crp.size
    
    def compute_determinism(self, crp: np.ndarray) -> float:
        """Compute determinism measure from CRP.
        
        Args:
            crp: Cross-recurrence plot matrix.
            
        Returns:
            Determinism measure.
        """
        # Find diagonal lines in CRP
        n, m = crp.shape
        diagonal_lengths = []
        
        # Check diagonals
        for k in range(-n + 1, m):
            diagonal = np.diagonal(crp, offset=k)
            if len(diagonal) > 0:
                # Find consecutive True values
                consecutive_length = 0
                max_consecutive = 0
                
                for val in diagonal:
                    if val:
                        consecutive_length += 1
                        max_consecutive = max(max_consecutive, consecutive_length)
                    else:
                        consecutive_length = 0
                
                if max_consecutive > 1:
                    diagonal_lengths.append(max_consecutive)
        
        if not diagonal_lengths:
            return 0.0
        
        # Determinism = sum of diagonal line lengths / total recurrent points
        total_recurrent = np.sum(crp)
        if total_recurrent == 0:
            return 0.0
        
        return sum(diagonal_lengths) / total_recurrent
