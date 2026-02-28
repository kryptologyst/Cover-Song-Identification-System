# Cover Song Identification System

## Privacy and Ethics Disclaimer

**IMPORTANT: This is a research and educational demonstration project only.**

- This system is designed for academic research and educational purposes
- It is NOT intended for production use or commercial applications
- The system does NOT store, process, or transmit personal identifiable information (PII)
- Any audio data processed is handled locally and not transmitted to external servers
- Users are responsible for ensuring they have proper rights to any audio content they process
- Misuse of this technology for unauthorized voice cloning, impersonation, or other malicious purposes is strictly prohibited
- The developers disclaim any responsibility for misuse of this technology

## Overview

This project implements a modern cover song identification system using advanced audio feature extraction and machine learning techniques. The system can identify whether two songs are cover versions of each other despite differences in performance style, instrumentation, and arrangement.

## Features

- **Advanced Audio Features**: MFCC, chroma, spectral contrast, tempo, rhythm patterns
- **Multiple Models**: Traditional ML (SVM, KNN) and deep learning approaches
- **DTW & CRP**: Dynamic Time Warping and Cross-Recurrence Plot analysis
- **Embedding Retrieval**: Triplet loss-based similarity learning
- **Comprehensive Evaluation**: mAP@k, R@k, DTW distance statistics
- **Interactive Demo**: Streamlit/Gradio interface for testing
- **Synthetic Dataset**: Built-in dataset generation for testing

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Cover-Song-Identification-System.git
cd Cover-Song-Identification-System

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Usage

```python
from src.models.cover_song_identifier import CoverSongIdentifier
from src.data.dataset import CoverSongDataset

# Initialize the model
identifier = CoverSongIdentifier()

# Load and process audio files
result = identifier.identify_cover_song("original.wav", "cover.wav")
print(f"Similarity score: {result['similarity']:.3f}")
print(f"Is cover: {result['is_cover']}")
```

### Running the Demo

```bash
# Streamlit demo
streamlit run demo/app.py

# Or Gradio demo
python demo/gradio_app.py
```

## Dataset Structure

The system expects audio files organized as follows:

```
data/
├── wav/                    # Audio files
│   ├── song1_original.wav
│   ├── song1_cover.wav
│   ├── song2_original.wav
│   └── song2_cover.wav
└── meta.csv               # Metadata file
    id,path,sr,duration,label,song_id,version
    song1_orig,data/wav/song1_original.wav,22050,180.5,original,song1,original
    song1_cov,data/wav/song1_cover.wav,22050,175.2,cover,song1,cover
```

## Training and Evaluation

### Training

```bash
python scripts/train.py --config configs/baseline.yaml
```

### Evaluation

```bash
python scripts/eval.py --model_path checkpoints/best_model.pth --test_data data/test/
```

## Configuration

The system uses Hydra/OmegaConf for configuration management. See `configs/` directory for example configurations.

## Metrics

The system evaluates performance using:

- **mAP@k**: Mean Average Precision at k
- **R@k**: Recall at k
- **DTW Distance**: Dynamic Time Warping distance statistics
- **Similarity Threshold**: ROC analysis for optimal threshold selection

## Model Architecture

### Traditional Approaches
- **SVM**: Support Vector Machine with RBF kernel
- **KNN**: K-Nearest Neighbors with distance weighting
- **Random Forest**: Ensemble method for robust classification

### Deep Learning Approaches
- **Siamese Networks**: Twin networks for similarity learning
- **Triplet Networks**: Embedding learning with triplet loss
- **Attention Mechanisms**: Focus on important audio segments

## Limitations

- Performance depends on audio quality and preprocessing
- May struggle with heavily modified covers (different genres, tempos)
- Requires sufficient training data for optimal performance
- Computational complexity increases with dataset size

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{cover_song_identification,
  title={Modern Cover Song Identification System},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Cover-Song-Identification-System}
}
```
# Cover-Song-Identification-System
