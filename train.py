import wandb
import torch
from transformers import GPT2Tokenizer, AutoTokenizer
import time
import pandas as pd

import io
from scipy.io.wavfile import write
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import librosa

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
import soundfile as sf
import IPython.display as ipd
import librosa
from IPython.display import display
from datasets import Dataset
from naturalspeech3_facodec.ns3_codec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download
from peft import get_peft_model, LoraConfig, TaskType
import torch





model = AutoModelForCausalLMWithValueHead.from_pretrained("amuvarma/luna-3days-tagged-noreps")
model = model.to("cuda")
def freeze_except_qkv(model):
    """
    Freezes all parameters in the model except for the Query, Key, and Value projection layers
    in the attention modules.
    
    Args:
        model: The LLaMA model with value head
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Then unfreeze only QKV layers
    for name, module in model.named_modules():
        if isinstance(module, type(model.pretrained_model.model.layers[0].self_attn)):
            # Unfreeze Q, K, V projection layers
            module.q_proj.weight.requires_grad = True
            module.k_proj.weight.requires_grad = True
            module.v_proj.weight.requires_grad = True
    
    # Print trainable parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

# Example usage:
freeze_except_qkv(model)
# print(model)

ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("amuvarma/luna-3days-tagged-noreps")


tkn = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tkn)
tokenizer.add_special_tokens(
    {'additional_special_tokens': [f"[T{i}]" for i in range(9000)]})

tokenizer.pad_token_id = 128263
tokenizer.pad_token = "[T7]"

ppo_config = {"mini_batch_size": 1, "batch_size": 1}
config = PPOConfig(**ppo_config)
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

csv_path = "./prompts_emotions_shuffled.csv"
df = pd.read_csv(csv_path)
queries = df["prompt"].tolist()
emotions = df["emotion"].tolist()


def generate_model_response(text, emotion):
    start_time = time.time()

    
    prompt = f'''<{emotion}> {text} </{emotion}>'''
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261]], dtype=torch.int64)

    # Concatenate the tensors
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

    input_ids = modified_input_ids  # Ensure input IDs are on the correct GPU
    attention_mask = torch.ones_like(input_ids)
    input_ids=  input_ids.to("cuda")

    generation_kwargs = {
        "pad_token_id": 128258,
        "top_k": 50,
        "temperature": 0.5,
        "repetition_penalty": 1.1,
        "max_length": 100,
        "eos_token_id": 128258
    }

    response_tensor = model.generate(input_ids, **generation_kwargs)

    print(f"Time taken to generate: {time.time() - start_time}")

    return input_ids, response_tensor


fa_encoder = FACodecEncoder(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
)

fa_decoder = FACodecDecoder(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
)

encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")

fa_encoder.load_state_dict(torch.load(encoder_ckpt))
fa_decoder.load_state_dict(torch.load(decoder_ckpt))

fa_encoder = fa_encoder
fa_decoder = fa_decoder

def process_audio_and_get_vq_id():

    test_wav_path = "spks.wav"
    test_wav = librosa.load(test_wav_path, sr=16000)[0]
    test_wav = torch.from_numpy(test_wav).float()
    test_wav = test_wav.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():

        # encode
        enc_out = fa_encoder(test_wav)
        enc_out = enc_out

        # quantize
        _, _, _, _, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)

        return spk_embs

spk_embs = process_audio_and_get_vq_id()


def decode_tensor(tensor_input):
    # Ensure the input is on the correct device
    vq_post_emb = fa_decoder.vq2emb(tensor_input)

    # Perform inference to get reconstructed waveform
    recon_wav = fa_decoder.inference(vq_post_emb, spk_embs)

    # Convert the audio to a NumPy array
    audio_samples = recon_wav[0][0].cpu().detach().numpy()

    # Optionally save the audio file
    sf.write("e_tok.wav", audio_samples, 16000)

    # Return the audio samples (and optionally play it)
    # ipd.display(ipd.Audio(audio_samples, rate=16000))

    return audio_samples



def convert_to_audio(generated_ids):
    
    sos_indices = (generated_ids[0] == 128000).nonzero(as_tuple=True)[0]
    second_sos_index = sos_indices[-1].item()

    eos_index = (generated_ids[0][second_sos_index:] == 128009).nonzero(as_tuple=True)[0][0].item() + second_sos_index

    extracted_tokens = generated_ids[0][second_sos_index : eos_index]

    decoded_text = tokenizer.decode(extracted_tokens)



    token_to_find = 128257
    token_to_remove = 128263

    # Check if the token exists in the tensor
    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()
        cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
    else:
        cropped_tensor = generated_ids

    mask = cropped_tensor != token_to_remove
    cropped_tensor = cropped_tensor[mask].view(cropped_tensor.size(0), -1)

    processed_tensor = cropped_tensor - 128266
    processed_tensor.shape



    original_shape = processed_tensor.shape
    new_dim_1 = (original_shape[1] // 6) * 6
    processed_tensor = processed_tensor[:, :new_dim_1]



    processed_tensor_content = processed_tensor[:, ::6]

    processed_tensor_prosody = processed_tensor[:, 1::6]
    processed_tensor_prosody = processed_tensor_prosody - 1024

    processed_tensor_content_1 = processed_tensor[:, 2::6]
    processed_tensor_content_1 = processed_tensor_content_1 - 2*1024

    processed_tensor_acoustic_1 = processed_tensor[:, 3::6]
    processed_tensor_acoustic_1 = processed_tensor_acoustic_1 - 3*1024

    processed_tensor_acoustic_2 = processed_tensor[:, 4::6]
    processed_tensor_acoustic_2 = processed_tensor_acoustic_2 - 4*1024

    processed_tensor_acoustic_3 = processed_tensor[:, 5::6]
    processed_tensor_acoustic_3 = processed_tensor_acoustic_3 - 5*1024


    stacked_tensor = torch.stack([processed_tensor_prosody, processed_tensor_content,processed_tensor_content_1, processed_tensor_acoustic_1,processed_tensor_acoustic_2, processed_tensor_acoustic_3, ], dim=0)

    test_0 = stacked_tensor[:, 0, :].unsqueeze(1)
    stacked_tensor = stacked_tensor.cpu()
    test_0 = test_0.cpu()

    audio_bits = decode_tensor(test_0)

    return audio_bits


vad_model = load_silero_vad()

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


def process_speech_timestamps(timestamps):
  if len(timestamps) == 0:
    return 0, 0
  elif len(timestamps) == 1:
    start_time = timestamps[0]['start']
    end_time = timestamps[0]['end']
  else:
    start_time = timestamps[0]['start']
    end_time = timestamps[-1]['end']

  return start_time, end_time


def preprocess_audio(audio_data, sample_rate=16000):
    if len(audio_data) == 0:
        return torch.zeros(1)

    audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
    wav_buffer = io.BytesIO()
    write(wav_buffer, sample_rate, audio_data)
    wav_buffer.seek(0)
    test_wav, _ = librosa.load(wav_buffer, sr=sample_rate)
    return torch.from_numpy(test_wav).float()



query=  "The old family photographs brought back the sweetest childhood memories."
emotion = "happy"

# for i, (query, emotion) in enumerate(zip(queries, emotions)):
# print(f"Processing query {i+1}/{len(queries)}: {query} with emotion {emotion}")


# ############## CODE TO SAVE GENERIC QUERY AND RESPONSE TENSORS ####################
# query_tensor, response_tensor = generate_model_response(query, emotion)

# torch.save(query_tensor, "query.pt")
# torch.save(response_tensor, "response.pt")

# ####################################################################################

############## CODE TO LOAD GENERIC QUERY AND RESPONSE TENSORS ####################

query_tensor = torch.load("query.pt")
response_tensor = torch.load("response.pt")

####################################################################################


audio = convert_to_audio(response_tensor)
audio_reward = get_reward_from_audio(audio )
reward = [torch.tensor(audio_reward, dtype=torch.float32)]  # Example fixed reward

train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
