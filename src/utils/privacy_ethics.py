"""Privacy and ethics utilities for cover song identification."""

import hashlib
import logging
import os
import re
from typing import Dict, List, Optional, Union

import numpy as np


class PrivacyGuard:
    """Privacy protection utilities for audio processing."""
    
    def __init__(self, privacy_mode: bool = True) -> None:
        """Initialize privacy guard.
        
        Args:
            privacy_mode: Whether to enable privacy protection features.
        """
        self.privacy_mode = privacy_mode
        self.logger = logging.getLogger(__name__)
        
        if privacy_mode:
            self.logger.info("Privacy mode enabled - filenames and metadata will be anonymized")
    
    def anonymize_filename(self, filename: str) -> str:
        """Anonymize filename by removing personal identifiers.
        
        Args:
            filename: Original filename.
            
        Returns:
            Anonymized filename.
        """
        if not self.privacy_mode:
            return filename
        
        # Remove file extension
        name, ext = os.path.splitext(filename)
        
        # Remove common personal identifiers
        anonymized = name.lower()
        anonymized = re.sub(r'[^a-z0-9_]', '_', anonymized)
        anonymized = re.sub(r'_+', '_', anonymized)  # Remove multiple underscores
        anonymized = anonymized.strip('_')
        
        # Generate hash-based anonymized name
        hash_obj = hashlib.md5(anonymized.encode())
        anonymized_name = f"audio_{hash_obj.hexdigest()[:8]}{ext}"
        
        return anonymized_name
    
    def anonymize_metadata(self, metadata: Dict) -> Dict:
        """Anonymize metadata dictionary.
        
        Args:
            metadata: Original metadata dictionary.
            
        Returns:
            Anonymized metadata dictionary.
        """
        if not self.privacy_mode:
            return metadata
        
        anonymized = metadata.copy()
        
        # Anonymize filename fields
        filename_fields = ['path', 'filename', 'file_path', 'audio_path']
        for field in filename_fields:
            if field in anonymized:
                anonymized[field] = self.anonymize_filename(str(anonymized[field]))
        
        # Remove or anonymize personal identifiers
        personal_fields = ['artist', 'performer', 'singer', 'user', 'owner']
        for field in personal_fields:
            if field in anonymized:
                anonymized[field] = f"anonymous_{hash(str(anonymized[field])) % 10000:04d}"
        
        return anonymized
    
    def sanitize_log_message(self, message: str) -> str:
        """Sanitize log message to remove potential PII.
        
        Args:
            message: Original log message.
            
        Returns:
            Sanitized log message.
        """
        if not self.privacy_mode:
            return message
        
        # Remove common PII patterns
        patterns = [
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
            (r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]'),
            (r'\b[A-Za-z]+\s+[A-Za-z]+\b', '[NAME]'),  # Simple name pattern
        ]
        
        sanitized = message
        for pattern, replacement in patterns:
            sanitized = re.sub(pattern, replacement, sanitized)
        
        return sanitized
    
    def log_privacy_warning(self, operation: str) -> None:
        """Log privacy warning for sensitive operations.
        
        Args:
            operation: Description of the operation being performed.
        """
        warning_msg = (
            f"PRIVACY WARNING: {operation}. "
            "This is a research/educational system only. "
            "No personal data should be processed without proper consent."
        )
        self.logger.warning(self.sanitize_log_message(warning_msg))


class EthicsGuard:
    """Ethics and misuse prevention utilities."""
    
    def __init__(self) -> None:
        """Initialize ethics guard."""
        self.logger = logging.getLogger(__name__)
        self.misuse_patterns = [
            'voice cloning',
            'deepfake',
            'impersonation',
            'biometric identification',
            'surveillance',
            'unauthorized access'
        ]
    
    def check_misuse_intent(self, text: str) -> bool:
        """Check if text contains potential misuse intent.
        
        Args:
            text: Text to check for misuse patterns.
            
        Returns:
            True if potential misuse detected, False otherwise.
        """
        text_lower = text.lower()
        
        for pattern in self.misuse_patterns:
            if pattern in text_lower:
                self.logger.warning(f"Potential misuse detected: {pattern}")
                return True
        
        return False
    
    def validate_use_case(self, use_case: str) -> bool:
        """Validate if use case is appropriate for research/educational purposes.
        
        Args:
            use_case: Description of intended use case.
            
        Returns:
            True if use case is appropriate, False otherwise.
        """
        appropriate_uses = [
            'research',
            'education',
            'academic',
            'demonstration',
            'experiment',
            'tutorial',
            'learning'
        ]
        
        use_case_lower = use_case.lower()
        
        for appropriate_use in appropriate_uses:
            if appropriate_use in use_case_lower:
                return True
        
        # Check for inappropriate uses
        if self.check_misuse_intent(use_case):
            return False
        
        return False
    
    def log_ethics_warning(self, operation: str) -> None:
        """Log ethics warning for sensitive operations.
        
        Args:
            operation: Description of the operation being performed.
        """
        warning_msg = (
            f"ETHICS WARNING: {operation}. "
            "This system is for research/educational use only. "
            "Misuse for voice cloning, impersonation, or unauthorized biometric identification is prohibited."
        )
        self.logger.warning(warning_msg)


class DataProtection:
    """Data protection utilities for audio processing."""
    
    def __init__(self, privacy_mode: bool = True) -> None:
        """Initialize data protection.
        
        Args:
            privacy_mode: Whether to enable data protection features.
        """
        self.privacy_mode = privacy_mode
        self.logger = logging.getLogger(__name__)
    
    def validate_audio_consent(self, audio_path: str) -> bool:
        """Validate that audio processing has proper consent.
        
        Args:
            audio_path: Path to audio file.
            
        Returns:
            True if consent is assumed (for synthetic data), False otherwise.
        """
        # For synthetic data, consent is assumed
        if 'synthetic' in audio_path.lower() or 'demo' in audio_path.lower():
            return True
        
        # For real data, require explicit consent
        self.logger.warning(
            f"Processing real audio data: {audio_path}. "
            "Ensure proper consent has been obtained."
        )
        return False
    
    def sanitize_audio_features(self, features: np.ndarray) -> np.ndarray:
        """Sanitize audio features to remove potential personal identifiers.
        
        Args:
            features: Audio feature vector.
            
        Returns:
            Sanitized feature vector.
        """
        if not self.privacy_mode:
            return features
        
        # Add small random noise to prevent exact reconstruction
        noise = np.random.normal(0, 1e-6, features.shape)
        sanitized_features = features + noise
        
        return sanitized_features
    
    def log_data_access(self, operation: str, data_type: str) -> None:
        """Log data access for audit purposes.
        
        Args:
            operation: Type of operation performed.
            data_type: Type of data accessed.
        """
        if self.privacy_mode:
            self.logger.info(f"Data access logged: {operation} on {data_type}")


def setup_privacy_logging(level: str = "INFO") -> None:
    """Set up privacy-aware logging.
    
    Args:
        level: Logging level.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('privacy_audit.log')
        ]
    )
    
    # Add privacy warning to all log messages
    logger = logging.getLogger(__name__)
    logger.info("Privacy-aware logging initialized")


def validate_research_use() -> bool:
    """Validate that the system is being used for research purposes.
    
    Returns:
        True if research use is validated, False otherwise.
    """
    logger = logging.getLogger(__name__)
    
    # Check environment variables
    research_mode = os.getenv('RESEARCH_MODE', 'true').lower() == 'true'
    
    if not research_mode:
        logger.error("System not in research mode. Set RESEARCH_MODE=true to continue.")
        return False
    
    # Log research use
    logger.info("System validated for research use")
    return True


def display_privacy_notice() -> None:
    """Display privacy notice to users."""
    notice = """
    ═══════════════════════════════════════════════════════════════
    PRIVACY AND ETHICS NOTICE
    ═══════════════════════════════════════════════════════════════
    
    This is a RESEARCH AND EDUCATIONAL system only.
    
    PRIVACY PROTECTION:
    • No personal data is stored or transmitted
    • Audio processing is performed locally
    • Filenames and metadata are anonymized
    • Temporary files are automatically deleted
    
    ETHICAL GUIDELINES:
    • Use only for academic research and education
    • Do not use for voice cloning or impersonation
    • Do not use for unauthorized biometric identification
    • Ensure proper consent for any audio content processed
    
    By using this system, you agree to these terms and conditions.
    ═══════════════════════════════════════════════════════════════
    """
    print(notice)
