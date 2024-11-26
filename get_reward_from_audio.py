import io
from scipy.io.wavfile import write
import numpy as np
import torch
import librosa
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


vad_model = load_silero_vad()




def preprocess_audio(audio_data, sample_rate=16000):
    if len(audio_data) == 0:
        return torch.zeros(1)

    audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
    wav_buffer = io.BytesIO()
    write(wav_buffer, sample_rate, audio_data)
    wav_buffer.seek(0)
    test_wav, _ = librosa.load(wav_buffer, sr=sample_rate)
    return torch.from_numpy(test_wav).float()



def get_reward_from_audio(audio, sample_rate=16000):
    audio_tensor = preprocess_audio(audio, sample_rate=sample_rate)


    speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, return_seconds=True)


    audio_len = len(audio) / sample_rate
    start_time, end_time = process_speech_timestamps(speech_timestamps)

    penalty = start_time/2
    if penalty > 1:
      penalty = 1

    reward = 1 - penalty

    if start_time == 0:
      reward = 0

    reward = round(reward, 2)

    return reward

