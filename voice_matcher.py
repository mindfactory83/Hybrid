import pickle
import numpy as np
import os
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from datetime import datetime

logger = logging.getLogger(__name__)

class VoiceMatcher:
    def __init__(self):
        self.threshold = 0.75  # Similarity threshold for authentication
        self.min_samples = 3   # Minimum samples required for enrollment
        
    def save_voice_sample(self, user_id, sample_number, features):
        """Save a voice sample for a user"""
        try:
            samples_dir = f"voiceprints/user_{user_id}_samples"
            os.makedirs(samples_dir, exist_ok=True)
            
            sample_path = os.path.join(samples_dir, f"sample_{sample_number}.pkl")
            
            sample_data = {
                'features': features,
                'timestamp': datetime.utcnow(),
                'sample_number': sample_number
            }
            
            with open(sample_path, 'wb') as f:
                pickle.dump(sample_data, f)
                
            logger.info(f"Saved voice sample {sample_number} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving voice sample: {str(e)}")
            return False
    
    def get_sample_count(self, user_id):
        """Get the number of voice samples for a user"""
        samples_dir = f"voiceprints/user_{user_id}_samples"
        if not os.path.exists(samples_dir):
            return 0
        
        sample_files = [f for f in os.listdir(samples_dir) if f.endswith('.pkl')]
        return len(sample_files)
    
    def create_voiceprint(self, user_id):
        """Create a consolidated voiceprint from multiple samples"""
        try:
            samples_dir = f"voiceprints/user_{user_id}_samples"
            if not os.path.exists(samples_dir):
                logger.error(f"No samples directory found for user {user_id}")
                return False
            
            # Load all samples
            samples = []
            for filename in os.listdir(samples_dir):
                if filename.endswith('.pkl'):
                    sample_path = os.path.join(samples_dir, filename)
                    with open(sample_path, 'rb') as f:
                        sample_data = pickle.load(f)
                        samples.append(sample_data['features'])
            
            if len(samples) < self.min_samples:
                logger.error(f"Insufficient samples for user {user_id}: {len(samples)}")
                return False
            
            # Create voiceprint by averaging samples and calculating statistics
            samples_array = np.array(samples)
            
            voiceprint = {
                'mean_features': np.mean(samples_array, axis=0),
                'std_features': np.std(samples_array, axis=0),
                'median_features': np.median(samples_array, axis=0),
                'sample_count': len(samples),
                'created_at': datetime.utcnow(),
                'raw_samples': samples_array  # Keep raw samples for additional matching
            }
            
            # Save the consolidated voiceprint
            voiceprint_path = f"voiceprints/user_{user_id}_voiceprint.pkl"
            with open(voiceprint_path, 'wb') as f:
                pickle.dump(voiceprint, f)
            
            logger.info(f"Created voiceprint for user {user_id} from {len(samples)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error creating voiceprint: {str(e)}")
            return False
    
    def authenticate_voice(self, user_id, test_features):
        """
        Authenticate a voice sample against stored voiceprint
        Returns tuple (is_match, confidence_score)
        """
        try:
            voiceprint_path = f"voiceprints/user_{user_id}_voiceprint.pkl"
            
            if not os.path.exists(voiceprint_path):
                logger.error(f"No voiceprint found for user {user_id}")
                return False, 0.0
            
            # Load stored voiceprint
            with open(voiceprint_path, 'rb') as f:
                voiceprint = pickle.load(f)
            
            # Multiple matching strategies for robust authentication
            scores = []
            
            # 1. Cosine similarity with mean features
            mean_similarity = cosine_similarity(
                test_features.reshape(1, -1),
                voiceprint['mean_features'].reshape(1, -1)
            )[0][0]
            scores.append(mean_similarity)
            
            # 2. Cosine similarity with median features
            median_similarity = cosine_similarity(
                test_features.reshape(1, -1),
                voiceprint['median_features'].reshape(1, -1)
            )[0][0]
            scores.append(median_similarity)
            
            # 3. Best match against individual samples
            best_sample_score = 0.0
            for sample in voiceprint['raw_samples']:
                sample_similarity = cosine_similarity(
                    test_features.reshape(1, -1),
                    sample.reshape(1, -1)
                )[0][0]
                best_sample_score = max(best_sample_score, sample_similarity)
            scores.append(best_sample_score)
            
            # 4. Statistical distance measure
            feature_diff = np.abs(test_features - voiceprint['mean_features'])
            normalized_diff = feature_diff / (voiceprint['std_features'] + 1e-8)
            stat_score = 1.0 / (1.0 + np.mean(normalized_diff))  # Convert distance to similarity
            scores.append(stat_score)
            
            # Combine scores with weights
            weights = [0.3, 0.2, 0.3, 0.2]  # Emphasize mean and best sample matches
            final_score = np.average(scores, weights=weights)
            
            # Decision based on threshold
            is_match = final_score >= self.threshold
            
            logger.info(f"Voice authentication for user {user_id}: score={final_score:.3f}, match={is_match}")
            logger.debug(f"Individual scores: mean={scores[0]:.3f}, median={scores[1]:.3f}, best_sample={scores[2]:.3f}, stat={scores[3]:.3f}")
            
            return is_match, final_score
            
        except Exception as e:
            logger.error(f"Error during voice authentication: {str(e)}")
            return False, 0.0
    
    def clear_user_voiceprint(self, user_id):
        """Clear all voice data for a user"""
        try:
            # Remove voiceprint file
            voiceprint_path = f"voiceprints/user_{user_id}_voiceprint.pkl"
            if os.path.exists(voiceprint_path):
                os.remove(voiceprint_path)
            
            # Remove samples directory
            samples_dir = f"voiceprints/user_{user_id}_samples"
            if os.path.exists(samples_dir):
                for filename in os.listdir(samples_dir):
                    os.remove(os.path.join(samples_dir, filename))
                os.rmdir(samples_dir)
            
            logger.info(f"Cleared voiceprint data for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing voiceprint: {str(e)}")
            return False
    
    def get_voiceprint_info(self, user_id):
        """Get information about a user's voiceprint"""
        try:
            voiceprint_path = f"voiceprints/user_{user_id}_voiceprint.pkl"
            
            if not os.path.exists(voiceprint_path):
                return None
            
            with open(voiceprint_path, 'rb') as f:
                voiceprint = pickle.load(f)
            
            return {
                'sample_count': voiceprint['sample_count'],
                'created_at': voiceprint['created_at'],
                'feature_dimensions': len(voiceprint['mean_features'])
            }
            
        except Exception as e:
            logger.error(f"Error getting voiceprint info: {str(e)}")
            return None
