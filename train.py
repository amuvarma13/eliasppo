import torch
from transformers import AutoTokenizer
import pandas as pd
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import torch
from transformers import AutoTokenizer
import librosa


from freeze_except_qkv import freeze_except_qkv
from generate_model_response import generate_model_response
from convert_to_audio import convert_to_audio
from get_reward_from_audio import get_reward_from_audio





model = AutoModelForCausalLMWithValueHead.from_pretrained("amuvarma/luna-3days-tagged-noreps")
model = model.to("cuda")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("amuvarma/luna-3days-tagged-noreps")

freeze_except_qkv(model)


tkn = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tkn)
tokenizer.add_special_tokens({'additional_special_tokens': [f"[T{i}]" for i in range(9000)]})
tokenizer.pad_token_id = 128263
tokenizer.pad_token = "[T7]"

ppo_config = {"mini_batch_size": 1, "batch_size": 1}
config = PPOConfig(**ppo_config)
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

csv_path = "./prompts_emotions_shuffled.csv"
df = pd.read_csv(csv_path)
queries = df["prompt"].tolist()
emotions = df["emotion"].tolist()

query=  "The old family photographs brought back the sweetest childhood memories."
emotion = "happy"

# for i, (query, emotion) in enumerate(zip(queries, emotions)):
# print(f"Processing query {i+1}/{len(queries)}: {query} with emotion {emotion}")


# ############## CODE TO SAVE GENERIC QUERY AND RESPONSE TENSORS ####################
query_tensor, response_tensor = generate_model_response(query, emotion, model, tokenizer)

torch.save(query_tensor, "query.pt")
torch.save(response_tensor, "response.pt")

# ####################################################################################




############## CODE TO LOAD GENERIC QUERY AND RESPONSE TENSORS ####################

# query_tensor = torch.load("query.pt")
# response_tensor = torch.load("response.pt")

####################################################################################

for i in range(100):
    audio = convert_to_audio(response_tensor, tokenizer)
    audio_reward = get_reward_from_audio(audio )
    reward = [torch.tensor(audio_reward, dtype=torch.float32)]  # Example fixed reward

    train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
    

