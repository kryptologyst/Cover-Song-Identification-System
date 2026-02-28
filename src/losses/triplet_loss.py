"""Triplet loss implementation for embedding-based cover song identification."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class TripletLoss(nn.Module):
    """Triplet loss for learning embeddings for cover song identification.
    
    The triplet loss encourages embeddings of cover songs to be closer to
    their original versions than to other songs in the embedding space.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        distance_metric: str = "euclidean",
        mining_strategy: str = "hard"
    ) -> None:
        """Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss.
            distance_metric: Distance metric ('euclidean', 'cosine').
            mining_strategy: Mining strategy ('hard', 'semi-hard', 'all').
        """
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.mining_strategy = mining_strategy
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim).
            positive: Positive embeddings (batch_size, embedding_dim).
            negative: Negative embeddings (batch_size, embedding_dim).
            
        Returns:
            Triplet loss value.
        """
        # Compute distances
        pos_dist = self._compute_distance(anchor, positive)
        neg_dist = self._compute_distance(anchor, negative)
        
        # Compute triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()
    
    def _compute_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Compute distance between embeddings.
        
        Args:
            x: First set of embeddings.
            y: Second set of embeddings.
            
        Returns:
            Distance tensor.
        """
        if self.distance_metric == "euclidean":
            return F.pairwise_distance(x, y, p=2)
        elif self.distance_metric == "cosine":
            return 1 - F.cosine_similarity(x, y)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")


class EmbeddingNetwork(nn.Module):
    """Neural network for learning audio embeddings.
    
    This network takes audio features as input and produces embeddings
    suitable for cover song identification using triplet loss.
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.2
    ) -> None:
        """Initialize embedding network.
        
        Args:
            input_dim: Input feature dimension.
            embedding_dim: Output embedding dimension.
            hidden_dims: Hidden layer dimensions.
            dropout: Dropout rate.
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.network = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input features (batch_size, input_dim).
            
        Returns:
            Embeddings (batch_size, embedding_dim).
        """
        return self.network(x)


class TripletMiner:
    """Mine triplets for triplet loss training.
    
    This class handles the mining of hard, semi-hard, or all possible
    triplets from a batch of embeddings and labels.
    """
    
    def __init__(self, mining_strategy: str = "hard") -> None:
        """Initialize triplet miner.
        
        Args:
            mining_strategy: Mining strategy ('hard', 'semi-hard', 'all').
        """
        self.mining_strategy = mining_strategy
    
    def mine_triplets(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mine triplets from embeddings and labels.
        
        Args:
            embeddings: Embeddings tensor (batch_size, embedding_dim).
            labels: Labels tensor (batch_size,).
            margin: Margin for triplet loss.
            
        Returns:
            Tuple of (anchor, positive, negative) embeddings.
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # Compute pairwise distances
        distances = self._compute_pairwise_distances(embeddings)
        
        # Find positive and negative pairs
        positive_mask = self._get_positive_mask(labels)
        negative_mask = self._get_negative_mask(labels)
        
        if self.mining_strategy == "hard":
            return self._mine_hard_triplets(
                embeddings, distances, positive_mask, negative_mask, margin
            )
        elif self.mining_strategy == "semi-hard":
            return self._mine_semi_hard_triplets(
                embeddings, distances, positive_mask, negative_mask, margin
            )
        elif self.mining_strategy == "all":
            return self._mine_all_triplets(
                embeddings, distances, positive_mask, negative_mask
            )
        else:
            raise ValueError(f"Unknown mining strategy: {self.mining_strategy}")
    
    def _compute_pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances between embeddings.
        
        Args:
            embeddings: Embeddings tensor.
            
        Returns:
            Distance matrix.
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        return distances
    
    def _get_positive_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """Get mask for positive pairs (same song, different versions).
        
        Args:
            labels: Labels tensor.
            
        Returns:
            Boolean mask for positive pairs.
        """
        batch_size = labels.size(0)
        mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=labels.device)
        
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    # Check if same song but different versions
                    # Assuming labels contain song_id information
                    song_i = labels[i] // 10  # Extract song ID
                    song_j = labels[j] // 10
                    version_i = labels[i] % 10  # Extract version
                    version_j = labels[j] % 10
                    
                    if song_i == song_j and version_i != version_j:
                        mask[i, j] = True
        
        return mask
    
    def _get_negative_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """Get mask for negative pairs (different songs).
        
        Args:
            labels: Labels tensor.
            
        Returns:
            Boolean mask for negative pairs.
        """
        batch_size = labels.size(0)
        mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=labels.device)
        
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    # Check if different songs
                    song_i = labels[i] // 10
                    song_j = labels[j] // 10
                    
                    if song_i != song_j:
                        mask[i, j] = True
        
        return mask
    
    def _mine_hard_triplets(
        self,
        embeddings: torch.Tensor,
        distances: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
        margin: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mine hard triplets.
        
        Hard triplets are those where the positive distance is larger than
        the negative distance plus margin.
        """
        anchors = []
        positives = []
        negatives = []
        
        batch_size = embeddings.size(0)
        
        for i in range(batch_size):
            # Find hardest positive
            pos_distances = distances[i][positive_mask[i]]
            if len(pos_distances) > 0:
                hardest_pos_idx = torch.argmax(pos_distances)
                pos_indices = torch.where(positive_mask[i])[0]
                hardest_pos = pos_indices[hardest_pos_idx]
                
                # Find hardest negative
                neg_distances = distances[i][negative_mask[i]]
                if len(neg_distances) > 0:
                    hardest_neg_idx = torch.argmin(neg_distances)
                    neg_indices = torch.where(negative_mask[i])[0]
                    hardest_neg = neg_indices[hardest_neg_idx]
                    
                    # Check if it's a valid hard triplet
                    pos_dist = distances[i, hardest_pos]
                    neg_dist = distances[i, hardest_neg]
                    
                    if pos_dist < neg_dist + margin:
                        anchors.append(embeddings[i])
                        positives.append(embeddings[hardest_pos])
                        negatives.append(embeddings[hardest_neg])
        
        if len(anchors) == 0:
            # Fallback to random triplets
            return self._mine_random_triplets(embeddings, positive_mask, negative_mask)
        
        return (
            torch.stack(anchors),
            torch.stack(positives),
            torch.stack(negatives)
        )
    
    def _mine_semi_hard_triplets(
        self,
        embeddings: torch.Tensor,
        distances: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
        margin: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mine semi-hard triplets.
        
        Semi-hard triplets are those where the negative distance is larger
        than the positive distance but smaller than positive distance + margin.
        """
        anchors = []
        positives = []
        negatives = []
        
        batch_size = embeddings.size(0)
        
        for i in range(batch_size):
            pos_distances = distances[i][positive_mask[i]]
            if len(pos_distances) > 0:
                pos_indices = torch.where(positive_mask[i])[0]
                
                for pos_idx in pos_indices:
                    pos_dist = distances[i, pos_idx]
                    
                    # Find semi-hard negatives
                    neg_distances = distances[i][negative_mask[i]]
                    neg_indices = torch.where(negative_mask[i])[0]
                    
                    for neg_idx in neg_indices:
                        neg_dist = distances[i, neg_idx]
                        
                        if pos_dist < neg_dist < pos_dist + margin:
                            anchors.append(embeddings[i])
                            positives.append(embeddings[pos_idx])
                            negatives.append(embeddings[neg_idx])
        
        if len(anchors) == 0:
            return self._mine_random_triplets(embeddings, positive_mask, negative_mask)
        
        return (
            torch.stack(anchors),
            torch.stack(positives),
            torch.stack(negatives)
        )
    
    def _mine_all_triplets(
        self,
        embeddings: torch.Tensor,
        distances: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mine all possible triplets."""
        anchors = []
        positives = []
        negatives = []
        
        batch_size = embeddings.size(0)
        
        for i in range(batch_size):
            pos_indices = torch.where(positive_mask[i])[0]
            neg_indices = torch.where(negative_mask[i])[0]
            
            for pos_idx in pos_indices:
                for neg_idx in neg_indices:
                    anchors.append(embeddings[i])
                    positives.append(embeddings[pos_idx])
                    negatives.append(embeddings[neg_idx])
        
        if len(anchors) == 0:
            return self._mine_random_triplets(embeddings, positive_mask, negative_mask)
        
        return (
            torch.stack(anchors),
            torch.stack(positives),
            torch.stack(negatives)
        )
    
    def _mine_random_triplets(
        self,
        embeddings: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mine random triplets as fallback."""
        anchors = []
        positives = []
        negatives = []
        
        batch_size = embeddings.size(0)
        
        for i in range(batch_size):
            pos_indices = torch.where(positive_mask[i])[0]
            neg_indices = torch.where(negative_mask[i])[0]
            
            if len(pos_indices) > 0 and len(neg_indices) > 0:
                pos_idx = pos_indices[torch.randint(len(pos_indices), (1,))]
                neg_idx = neg_indices[torch.randint(len(neg_indices), (1,))]
                
                anchors.append(embeddings[i])
                positives.append(embeddings[pos_idx])
                negatives.append(embeddings[neg_idx])
        
        if len(anchors) == 0:
            # Return empty tensors
            return (
                torch.empty(0, embeddings.size(1), device=embeddings.device),
                torch.empty(0, embeddings.size(1), device=embeddings.device),
                torch.empty(0, embeddings.size(1), device=embeddings.device)
            )
        
        return (
            torch.stack(anchors),
            torch.stack(positives),
            torch.stack(negatives)
        )
