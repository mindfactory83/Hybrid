
import pickle
import numpy as np
import os
import logging
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class VoiceMatcher:
    def __init__(self):
        self.threshold = 0.75

    def save_voice_sample(self, user_id, sample_number, embedding):
        try:
            samples_dir = f"voiceprints/user_{user_id}_samples"
            os.makedirs(samples_dir, exist_ok=True)
            path = os.path.join(samples_dir, f"sample_{sample_number}.pkl")
            data = {
                'embedding': embedding,
                'timestamp': datetime.utcnow()
            }
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved embedding sample {sample_number} for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving voice sample: {str(e)}")
            return False

    def get_sample_count(self, user_id):
        path = f"voiceprints/user_{user_id}_samples"
        return len([f for f in os.listdir(path) if f.endswith('.pkl')]) if os.path.exists(path) else 0

    def create_voiceprint(self, user_id):
        try:
            path = f"voiceprints/user_{user_id}_samples"
            if not os.path.exists(path):
                logger.error("Samples path not found.")
                return False
            embeddings = []
            for file in os.listdir(path):
                if file.endswith('.pkl'):
                    with open(os.path.join(path, file), 'rb') as f:
                        data = pickle.load(f)
                        embeddings.append(data['embedding'])
            if len(embeddings) < 3:
                logger.warning(f"Not enough samples to create voiceprint for user {user_id}")
                return False
            voiceprint = {
                'embedding': np.mean(embeddings, axis=0)
            }
            with open(f"voiceprints/user_{user_id}_voiceprint.pkl", 'wb') as f:
                pickle.dump(voiceprint, f)
            logger.info(f"Created voiceprint for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating voiceprint: {str(e)}")
            return False

    def authenticate_voice(self, user_id, test_embedding):
        try:
            path = f"voiceprints/user_{user_id}_voiceprint.pkl"
            if not os.path.exists(path):
                return False, 0.0
            with open(path, 'rb') as f:
                vp = pickle.load(f)

            sim = cosine_similarity(
                [vp['embedding']],
                [test_embedding]
            )[0][0]

            is_match = sim >= self.threshold
            logger.info(f"Cosine similarity: {sim:.3f}, Match: {is_match}")
            return is_match, sim
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False, 0.0

    def clear_user_voiceprint(self, user_id):
        try:
            vp_path = f"voiceprints/user_{user_id}_voiceprint.pkl"
            if os.path.exists(vp_path): os.remove(vp_path)
            samples_dir = f"voiceprints/user_{user_id}_samples"
            if os.path.exists(samples_dir):
                for f in os.listdir(samples_dir):
                    os.remove(os.path.join(samples_dir, f))
                os.rmdir(samples_dir)
            logger.info(f"Cleared voiceprint for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing voiceprint: {str(e)}")
            return False
