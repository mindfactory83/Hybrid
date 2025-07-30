import librosa
import numpy as np
import logging
from scipy import signal
import os

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 16000  # Standard sample rate for voice
        self.n_mfcc = 13  # Number of MFCC coefficients
        self.n_fft = 2048
        self.hop_length = 512
        
    def extract_mfcc_features(self, audio_path):
        """
        Extract MFCC features from audio file
        Returns normalized MFCC features or None if processing fails
        """
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Validate audio
            if len(audio) == 0:
                logger.error("Empty audio file")
                return None
                
            # Apply pre-emphasis filter
            audio = self._pre_emphasis(audio)
            
            # Remove silence
            audio = self._remove_silence(audio)
            
            if len(audio) < 0.5 * self.sample_rate:  # Less than 0.5 seconds
                logger.error("Audio too short after silence removal")
                return None
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Normalize features
            mfccs = self._normalize_features(mfccs)
            
            # Calculate delta and delta-delta features
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            
            # Combine all features
            features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
            
            # Calculate statistical features (mean, std, min, max)
            feature_stats = self._calculate_statistical_features(features)
            
            logger.info(f"Extracted features shape: {feature_stats.shape}")
            return feature_stats
            
        except Exception as e:
            logger.error(f"Error extracting MFCC features: {str(e)}")
            return None
    
    def _pre_emphasis(self, audio, alpha=0.97):
        """Apply pre-emphasis filter to enhance high frequencies"""
        return np.append(audio[0], audio[1:] - alpha * audio[:-1])
    
    def _remove_silence(self, audio, top_db=20):
        """Remove silence from audio using librosa's trim function"""
        try:
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            return audio_trimmed
        except:
            return audio
    
    def _normalize_features(self, features):
        """Normalize MFCC features to zero mean and unit variance"""
        return (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + 1e-8)
    
    def _calculate_statistical_features(self, features):
        """Calculate statistical features from MFCC coefficients"""
        # Calculate statistics along time axis
        mean_features = np.mean(features, axis=1)
        std_features = np.std(features, axis=1)
        min_features = np.min(features, axis=1)
        max_features = np.max(features, axis=1)
        
        # Combine all statistical features
        combined_features = np.concatenate([
            mean_features,
            std_features,
            min_features,
            max_features
        ])
        
        return combined_features
    
    def validate_audio_quality(self, audio_path):
        """
        Validate audio quality for voice authentication
        Returns tuple (is_valid, error_message)
        """
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Check duration
            duration = len(audio) / sr
            if duration < 1.0:
                return False, "Audio too short (minimum 1 second required)"
            if duration > 10.0:
                return False, "Audio too long (maximum 10 seconds allowed)"
            
            # Check signal level
            rms = np.sqrt(np.mean(audio**2))
            if rms < 0.01:
                return False, "Audio signal too weak"
            
            # Check for clipping
            if np.max(np.abs(audio)) > 0.99:
                return False, "Audio signal is clipped"
            
            # Check sample rate
            if sr < 8000:
                return False, "Sample rate too low (minimum 8kHz)"
            
            return True, "Audio quality is acceptable"
            
        except Exception as e:
            return False, f"Error validating audio: {str(e)}"
