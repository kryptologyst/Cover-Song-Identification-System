"""Streamlit demo application for cover song identification."""

import os
import tempfile
from typing import Dict, Optional, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.data.synthetic_dataset import SyntheticCoverSongDataset
from src.features.audio_features import AudioFeatureExtractor
from src.metrics.evaluation import CoverSongMetrics
from src.models.baseline import BaselineCoverSongIdentifier
from src.utils.device import get_device, anonymize_filename


# Configure Streamlit page
st.set_page_config(
    page_title="Cover Song Identification Demo",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Privacy disclaimer
st.markdown("""
<div style="background-color: #ffebee; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
<h4 style="color: #c62828; margin-top: 0;">Privacy and Ethics Disclaimer</h4>
<p style="margin-bottom: 0;">
<strong>This is a research and educational demonstration only.</strong><br>
• No personal data is stored or transmitted<br>
• Audio processing is performed locally<br>
• Misuse for unauthorized voice cloning is prohibited<br>
• Users must have proper rights to any audio content processed
</p>
</div>
""", unsafe_allow_html=True)

# Title and description
st.title("🎵 Cover Song Identification System")
st.markdown("""
This demo showcases a modern cover song identification system using advanced audio features 
and machine learning techniques. Upload two audio files to determine if one is a cover version of the other.
""")

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Model Type",
    ["baseline", "dtw", "crp", "combined"],
    help="Choose the identification method"
)

# Similarity threshold
threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Threshold for cover song classification"
)

# Privacy settings
privacy_mode = st.sidebar.checkbox(
    "Privacy Mode",
    value=True,
    help="Anonymize filenames and metadata"
)

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
if "feature_extractor" not in st.session_state:
    st.session_state.feature_extractor = None
if "dataset" not in st.session_state:
    st.session_state.dataset = None


@st.cache_resource
def load_model():
    """Load the cover song identification model."""
    device = get_device("cpu")  # Use CPU for demo
    
    feature_extractor = AudioFeatureExtractor()
    model = BaselineCoverSongIdentifier(
        feature_extractor=feature_extractor,
        classifier_type="svm",
        similarity_threshold=threshold,
        device=device
    )
    
    return model, feature_extractor


@st.cache_resource
def load_synthetic_dataset():
    """Load or generate synthetic dataset for testing."""
    dataset = SyntheticCoverSongDataset(
        data_dir="data",
        num_songs=50,
        num_covers_per_song=2,
        privacy_mode=privacy_mode
    )
    
    # Generate dataset if it doesn't exist
    if not os.path.exists(os.path.join("data", "meta.csv")):
        st.info("Generating synthetic dataset... This may take a moment.")
        dataset.generate_dataset()
    
    return dataset


def load_model_and_dataset():
    """Load model and dataset."""
    if st.session_state.model is None:
        st.session_state.model, st.session_state.feature_extractor = load_model()
    
    if st.session_state.dataset is None:
        st.session_state.dataset = load_synthetic_dataset()


def extract_audio_features(audio_file) -> np.ndarray:
    """Extract features from uploaded audio file."""
    try:
        # Load audio
        audio, sr = librosa.load(audio_file, sr=22050)
        
        # Extract features
        features = st.session_state.feature_extractor.extract_combined_features(audio)
        
        return features, audio, sr
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None, None, None


def plot_audio_waveform(audio: np.ndarray, sr: int, title: str) -> None:
    """Plot audio waveform."""
    fig, ax = plt.subplots(figsize=(10, 3))
    
    time = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(time, audio)
    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    
    st.pyplot(fig)
    plt.close()


def plot_spectrogram(audio: np.ndarray, sr: int, title: str) -> None:
    """Plot audio spectrogram."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Compute spectrogram
    stft = librosa.stft(audio)
    magnitude = np.abs(stft)
    
    # Convert to dB
    log_magnitude = librosa.amplitude_to_db(magnitude)
    
    # Plot
    img = librosa.display.specshow(
        log_magnitude,
        sr=sr,
        x_axis="time",
        y_axis="hz",
        ax=ax
    )
    
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    
    st.pyplot(fig)
    plt.close()


def compare_features(features1: np.ndarray, features2: np.ndarray) -> None:
    """Compare feature vectors visually."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot individual features
    ax1.plot(features1, label="Audio 1", alpha=0.7)
    ax1.plot(features2, label="Audio 2", alpha=0.7)
    ax1.set_title("Feature Comparison")
    ax1.set_xlabel("Feature Index")
    ax1.set_ylabel("Feature Value")
    ax1.legend()
    ax1.grid(True)
    
    # Plot feature differences
    feature_diff = np.abs(features1 - features2)
    ax2.plot(feature_diff)
    ax2.set_title("Feature Differences")
    ax2.set_xlabel("Feature Index")
    ax2.set_ylabel("Absolute Difference")
    ax2.grid(True)
    
    # Plot correlation
    correlation = np.corrcoef(features1, features2)[0, 1]
    ax3.bar(["Correlation"], [correlation])
    ax3.set_title(f"Feature Correlation: {correlation:.3f}")
    ax3.set_ylabel("Correlation Coefficient")
    ax3.set_ylim(-1, 1)
    
    st.pyplot(fig)
    plt.close()


def main():
    """Main demo application."""
    # Load model and dataset
    load_model_and_dataset()
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Upload Audio Files")
        
        # File upload
        audio_file1 = st.file_uploader(
            "Upload First Audio File",
            type=["wav", "mp3", "flac", "m4a"],
            help="Upload the first audio file to compare"
        )
        
        audio_file2 = st.file_uploader(
            "Upload Second Audio File",
            type=["wav", "mp3", "flac", "m4a"],
            help="Upload the second audio file to compare"
        )
        
        # Or use synthetic examples
        st.subheader("Or Try Synthetic Examples")
        if st.button("Load Random Synthetic Pair"):
            dataset = st.session_state.dataset
            pairs = dataset.get_cover_pairs()
            
            if pairs:
                pair = np.random.choice(len(pairs))
                audio_file1_path, audio_file2_path = pairs[pair]
                
                # Create file-like objects
                with open(audio_file1_path, "rb") as f:
                    audio_file1 = f.read()
                with open(audio_file2_path, "rb") as f:
                    audio_file2 = f.read()
                
                st.success("Loaded synthetic audio pair!")
    
    with col2:
        st.header("Analysis Results")
        
        if audio_file1 is not None and audio_file2 is not None:
            # Process audio files
            with st.spinner("Processing audio files..."):
                # Extract features
                features1, audio1, sr1 = extract_audio_features(audio_file1)
                features2, audio2, sr2 = extract_audio_features(audio_file2)
                
                if features1 is not None and features2 is not None:
                    # Perform cover song identification
                    model = st.session_state.model
                    
                    # Create temporary files for the model
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp1:
                        tmp1.write(audio_file1.read() if hasattr(audio_file1, 'read') else audio_file1)
                        tmp1_path = tmp1.name
                    
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
                        tmp2.write(audio_file2.read() if hasattr(audio_file2, 'read') else audio_file2)
                        tmp2_path = tmp2.name
                    
                    try:
                        # Get identification result
                        result = model.identify_cover_song(tmp1_path, tmp2_path, method=model_type)
                        
                        # Display results
                        similarity = result["similarity"]
                        is_cover = result["is_cover"]
                        
                        # Similarity score
                        st.metric(
                            "Similarity Score",
                            f"{similarity:.3f}",
                            delta=f"{similarity - threshold:.3f}" if similarity > threshold else f"{similarity - threshold:.3f}"
                        )
                        
                        # Cover song prediction
                        if is_cover:
                            st.success("🎵 **Cover Song Detected!**")
                            st.info("The two audio files appear to be cover versions of each other.")
                        else:
                            st.warning("❌ **Not a Cover Song**")
                            st.info("The two audio files do not appear to be cover versions of each other.")
                        
                        # Additional metrics if available
                        if "classifier_similarity" in result:
                            st.subheader("Detailed Analysis")
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Classifier Similarity", f"{result['classifier_similarity']:.3f}")
                            with col_b:
                                st.metric("DTW Similarity", f"{result['dtw_similarity']:.3f}")
                            with col_c:
                                st.metric("CRP Similarity", f"{result['crp_similarity']:.3f}")
                        
                    finally:
                        # Clean up temporary files
                        os.unlink(tmp1_path)
                        os.unlink(tmp2_path)
                    
                    # Audio visualization
                    st.subheader("Audio Visualization")
                    
                    tab1, tab2, tab3 = st.tabs(["Waveforms", "Spectrograms", "Feature Comparison"])
                    
                    with tab1:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            plot_audio_waveform(audio1, sr1, "Audio File 1")
                        with col_b:
                            plot_audio_waveform(audio2, sr2, "Audio File 2")
                    
                    with tab2:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            plot_spectrogram(audio1, sr1, "Audio File 1 Spectrogram")
                        with col_b:
                            plot_spectrogram(audio2, sr2, "Audio File 2 Spectrogram")
                    
                    with tab3:
                        compare_features(features1, features2)
                    
                    # Audio playback
                    st.subheader("Audio Playback")
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.audio(audio_file1, format="audio/wav")
                        if privacy_mode:
                            filename1 = anonymize_filename("audio_file_1")
                        else:
                            filename1 = "Audio File 1"
                        st.caption(filename1)
                    
                    with col_b:
                        st.audio(audio_file2, format="audio/wav")
                        if privacy_mode:
                            filename2 = anonymize_filename("audio_file_2")
                        else:
                            filename2 = "Audio File 2"
                        st.caption(filename2)
                
                else:
                    st.error("Failed to process one or both audio files.")
        
        else:
            st.info("Please upload two audio files to perform cover song identification.")
    
    # Dataset information
    st.sidebar.header("Dataset Information")
    if st.session_state.dataset is not None:
        dataset = st.session_state.dataset
        try:
            df = dataset.load_dataset()
            
            st.sidebar.metric("Total Files", len(df))
            st.sidebar.metric("Original Songs", len(df[df["label"] == "original"]))
            st.sidebar.metric("Cover Songs", len(df[df["label"] == "cover"]))
            
            # Show sample data
            if st.sidebar.checkbox("Show Sample Data"):
                st.sidebar.dataframe(df.head(10))
        
        except FileNotFoundError:
            st.sidebar.info("Dataset not found. Click 'Generate Dataset' to create one.")
    
    # Technical details
    with st.expander("Technical Details"):
        st.markdown("""
        **Feature Extraction:**
        - MFCC (Mel-frequency cepstral coefficients)
        - Chroma features
        - Spectral contrast
        - Tempo and rhythm patterns
        - Zero crossing rate
        - Spectral centroid and rolloff
        
        **Identification Methods:**
        - **Baseline**: Traditional ML classifiers (SVM, KNN, Random Forest)
        - **DTW**: Dynamic Time Warping for sequence alignment
        - **CRP**: Cross-Recurrence Plot for pattern analysis
        - **Combined**: Weighted combination of all methods
        
        **Evaluation Metrics:**
        - mAP@k (Mean Average Precision at k)
        - R@k (Recall at k)
        - ROC AUC and Average Precision
        - DTW distance statistics
        """)


if __name__ == "__main__":
    main()
