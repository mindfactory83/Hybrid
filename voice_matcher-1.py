
import pickle
import numpy as np
import os
import logging
from datetime import datetime
from librosa.sequence import dtw

logger = logging.getLogger(__name__)

class VoiceMatcher:
    def __init__(self):
        self.threshold = 0.1

    def dtw_distance(self, A, B):
        try:
            A = A.T if A.shape[0] < A.shape[1] else A
            B = B.T if B.shape[0] < B.shape[1] else B
            logger.info(f"DTW input shapes: test={A.shape}, enrolled={B.shape}")
            D, _ = dtw(A.T, B.T, metric='euclidean')
            dist = D[-1, -1]
            logger.info(f"DTW raw distance: {dist:.2f}")
            return dist
        except Exception as e:
            logger.error(f"DTW error: {str(e)}")
            return float('inf')

    def save_voice_sample(self, user_id, sample_number, mfcc_matrix):
        try:
            samples_dir = f"voiceprints/user_{user_id}_samples"
            os.makedirs(samples_dir, exist_ok=True)
            sample_path = os.path.join(samples_dir, f"sample_{sample_number}.pkl")
            data = {
                'features': mfcc_matrix,
                'timestamp': datetime.utcnow()
            }
            with open(sample_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved voice sample {sample_number} for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving voice sample: {str(e)}")
            return False

    def get_sample_count(self, user_id):
        path = f"voiceprints/user_{user_id}_samples"
        if not os.path.exists(path):
            return 0
        return len([f for f in os.listdir(path) if f.endswith('.pkl')])

    def create_voiceprint(self, user_id):
        try:
            path = f"voiceprints/user_{user_id}_samples"
            if not os.path.exists(path):
                logger.error("Samples path not found.")
                return False
            samples = []
            for file in os.listdir(path):
                if file.endswith('.pkl'):
                    with open(os.path.join(path, file), 'rb') as f:
                        data = pickle.load(f)
                        samples.append(data['features'])
            if len(samples) < 3:
                return False
            with open(f"voiceprints/user_{user_id}_voiceprint.pkl", 'wb') as f:
                pickle.dump(samples, f)
            return True
        except Exception as e:
            logger.error(f"Error creating voiceprint: {str(e)}")
            return False

    def authenticate_voice(self, user_id, test_matrix):
        try:
            path = f"voiceprints/user_{user_id}_voiceprint.pkl"
            if not os.path.exists(path):
                return False, 0.0
            with open(path, 'rb') as f:
                enrolled_samples = pickle.load(f)
            distances = [self.dtw_distance(test_matrix, enrolled) for enrolled in enrolled_samples]
            avg_dist = np.mean(distances)
            #similarity = 1.0 / (1.0 + avg_dist)
            #similarity = 0.0 if not np.isfinite(avg_dist) else 1.0 / (1.0 + avg_dist)

            alpha = 40.0
            similarity = np.exp(-avg_dist / alpha)
            
            is_match = similarity >= self.threshold
            logger.info(f"DTW voice match score={similarity:.3f}, match={is_match}")
            return is_match, similarity
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False, 0.0
    def clear_user_voiceprint(self, user_id):
        try:
            vp_path = f"voiceprints/user_{user_id}_voiceprint.pkl"
            if os.path.exists(vp_path):
                os.remove(vp_path)
            samples_dir = f"voiceprints/user_{user_id}_samples"
            if os.path.exists(samples_dir):
                for f in os.listdir(samples_dir):
                    os.remove(os.path.join(samples_dir, f))
                os.rmdir(samples_dir)
            logger.info(f"Cleared voiceprint data for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing voiceprint: {str(e)}")
            return False
    
    def get_voiceprint_info(self, user_id):
        try:
            path = f"voiceprints/user_{user_id}_voiceprint.pkl"
            if not os.path.exists(path):
                return None
            with open(path, 'rb') as f:
                vp = pickle.load(f)
            return {
                'sample_count': vp['sample_count'],
                'created_at': vp['created_at'],
                'feature_dimensions': len(vp['mean_features'])
            }
        except Exception as e:
            logger.error(f"Error getting voiceprint info: {str(e)}")
            return None

