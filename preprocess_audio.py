import io
from scipy.io.wavfile import write
import numpy as np
import torch
import librosa



def preprocess_audio(audio_data, sample_rate=16000):
    if len(audio_data) == 0:
        return torch.zeros(1)

    audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
    wav_buffer = io.BytesIO()
    write(wav_buffer, sample_rate, audio_data)
    wav_buffer.seek(0)
    test_wav, _ = librosa.load(wav_buffer, sr=sample_rate)
    return torch.from_numpy(test_wav).float()

