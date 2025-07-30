
import torchaudio
import torch
import logging
import os
import numpy as np
from speechbrain.pretrained import SpeakerRecognition

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 16000
        self.speaker_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/ecapa"
        )

    def extract_embedding(self, audio_path):
        try:
            signal, sr = torchaudio.load(audio_path)
            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)
            embedding = self.speaker_model.encode_batch(signal).squeeze().detach().cpu().numpy()
            return embedding
        except Exception as e:
            logger.error(f"ECAPA embedding error: {str(e)}")
            return None
