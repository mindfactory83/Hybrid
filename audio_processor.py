
import librosa
import numpy as np
import logging
from scipy import signal
import os

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 16000  # Standard sample rate for voice
        self.n_mfcc = 20  # Increased MFCCs for better discrimination
        self.n_fft = 2048
        self.hop_length = 512
        
    def extract_mfcc_features(self, audio_path):
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            if len(audio) == 0:
                logger.error("Empty audio file")
                return None
                
            audio = self._pre_emphasis(audio)
            audio = self._remove_silence(audio)
            
            if len(audio) < 0.5 * self.sample_rate:
                logger.error("Audio too short after silence removal")
                return None
            
            # MFCC + deltas + spectral
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
            delta = librosa.feature.delta(mfccs)
            delta2 = librosa.feature.delta(mfccs, order=2)
            zcr = librosa.feature.zero_crossing_rate(y=audio)
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            
            features = np.vstack([mfccs, delta, delta2, zcr, centroid, bandwidth])
            
            #features = self._normalize_features(features)
            logger.info(f"MFCC shape before Norm: {mfccs.shape}, mean: {np.mean(mfccs):.2f}, std: {np.std(mfccs):.2f}")

            mfccs = self._normalize_mfcc(mfccs)
            # Statistical features
            mean_features = np.mean(features, axis=1)
            std_features = np.std(features, axis=1)
            min_features = np.min(features, axis=1)
            max_features = np.max(features, axis=1)
            
            combined_features = np.concatenate([mean_features, std_features, min_features, max_features])
            #combined_features =self._normalize_features(combined_features)
            logger.info(f"Extracted features shape: {combined_features.shape}")
            #return combined_features
            
            logger.info(f"MFCC shape: {mfccs.shape}")  # should be (20, T)
            # Log duration and stats
            duration = librosa.get_duration(y=audio, sr=sr)
            logger.info(f"Audio duration: {duration:.2f}s")
            logger.info(f"MFCC shape: {mfccs.shape}, mean: {np.mean(mfccs):.2f}, std: {np.std(mfccs):.2f}")

            return {"matrix": mfccs, "stats": combined_features}     
            
        except Exception as e:
            logger.error(f"Error extracting MFCC features: {str(e)}")
            return None
    
    def _pre_emphasis(self, audio, alpha=0.97):
        return np.append(audio[0], audio[1:] - alpha * audio[:-1])
    
    def _remove_silence(self, audio, top_db=20):
        try:
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            return audio_trimmed
        except:
            return audio
    
    def _normalize_features(self, features):
        return (features - np.mean(features)) / (np.std(features) + 1e-8)

    def _normalize_mfcc(self, mfcc):
        return (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

    def _calculate_statistical_features(self, features):
        mean_features = np.mean(features, axis=1)
        std_features = np.std(features, axis=1)
        min_features = np.min(features, axis=1)
        max_features = np.max(features, axis=1)
        return np.concatenate([mean_features, std_features, min_features, max_features])
    
    def validate_audio_quality(self, audio_path):
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            duration = len(audio) / sr
            if duration < 1.0:
                return False, "Audio too short (minimum 1 second required)"
            if duration > 10.0:
                return False, "Audio too long (maximum 10 seconds allowed)"
            rms = np.sqrt(np.mean(audio**2))
            if rms < 0.01:
                return False, "Audio signal too weak"
            if np.max(np.abs(audio)) > 0.99:
                return False, "Audio signal is clipped"
            if sr < 8000:
                return False, "Sample rate too low (minimum 8kHz)"
            return True, "Audio quality is acceptable"
        except Exception as e:
            return False, f"Error validating audio: {str(e)}"
