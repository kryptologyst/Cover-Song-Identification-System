#!/usr/bin/env python3
"""Quick start script for cover song identification demo."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.synthetic_dataset import SyntheticCoverSongDataset
from src.features.audio_features import AudioFeatureExtractor
from src.models.baseline import BaselineCoverSongIdentifier
from src.utils.device import set_seed


def main():
    """Quick start demonstration."""
    print("🎵 Cover Song Identification - Quick Start Demo")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Step 1: Generate synthetic dataset
    print("\n1. Generating synthetic dataset...")
    dataset = SyntheticCoverSongDataset(
        data_dir="data",
        num_songs=20,
        num_covers_per_song=2,
        audio_duration=10.0,  # 10 seconds for quick demo
        privacy_mode=True
    )
    
    if not os.path.exists("data/meta.csv"):
        df = dataset.generate_dataset()
        print(f"   Generated {len(df)} audio files")
        print(f"   Original songs: {len(df[df['label'] == 'original'])}")
        print(f"   Cover songs: {len(df[df['label'] == 'cover'])}")
    else:
        df = dataset.load_dataset()
        print(f"   Using existing dataset with {len(df)} files")
    
    # Step 2: Initialize feature extractor
    print("\n2. Initializing feature extractor...")
    feature_extractor = AudioFeatureExtractor()
    print(f"   Feature dimension: {feature_extractor.get_feature_dimension()}")
    
    # Step 3: Initialize model
    print("\n3. Initializing cover song identifier...")
    model = BaselineCoverSongIdentifier(
        feature_extractor=feature_extractor,
        classifier_type="svm",
        similarity_threshold=0.5
    )
    
    # Step 4: Test with synthetic examples
    print("\n4. Testing with synthetic examples...")
    pairs = dataset.get_cover_pairs()
    
    if pairs:
        # Test first few pairs
        test_pairs = pairs[:3]
        
        for i, (original_path, cover_path) in enumerate(test_pairs):
            print(f"\n   Test {i+1}:")
            print(f"   Original: {os.path.basename(original_path)}")
            print(f"   Cover: {os.path.basename(cover_path)}")
            
            try:
                result = model.identify_cover_song(original_path, cover_path, method="combined")
                
                print(f"   Similarity: {result['similarity']:.3f}")
                print(f"   Is Cover: {'Yes' if result['is_cover'] else 'No'}")
                
                if "classifier_similarity" in result:
                    print(f"   Classifier: {result['classifier_similarity']:.3f}")
                    print(f"   DTW: {result['dtw_similarity']:.3f}")
                    print(f"   CRP: {result['crp_similarity']:.3f}")
                
            except Exception as e:
                print(f"   Error: {e}")
    
    # Step 5: Show next steps
    print("\n5. Next Steps:")
    print("   • Run the Streamlit demo: streamlit run demo/app.py")
    print("   • Train a model: python scripts/train.py")
    print("   • Evaluate model: python scripts/eval.py --model_path checkpoints/best_model.pkl")
    print("   • Run tests: python -m pytest tests/")
    
    print("\n" + "=" * 50)
    print("Demo completed! Check the 'data' directory for generated audio files.")
    print("Privacy Note: All filenames have been anonymized for privacy protection.")


if __name__ == "__main__":
    main()
