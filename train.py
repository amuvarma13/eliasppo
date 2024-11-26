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





model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "amuvarma/luna-3days-tagged-noreps",
    # torch_dtype=torch.float16
)

model = model.to("cuda")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("amuvarma/luna-3days-tagged-noreps", 
                                                    
    # torch_dtype=torch.float16
)

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


max_tokens = 400  #num toks generated to analyse
log_steps = 4


for i, (query, emotion) in enumerate(zip(queries, emotions)):
    print(f"processing sample {i}")

    query_tensor, response_tensor = generate_model_response(query, emotion, ppo_trainer.model.module, tokenizer, max_length=max_tokens)
    #use ppotrainer.model instead of model to get the updated model (otherwise it will be the same as the ref model) (*i think)

    audio = convert_to_audio(response_tensor, tokenizer)
    audio_reward = get_reward_from_audio(audio )
    reward = [torch.tensor(audio_reward, dtype=torch.float32)]  # Example fixed reward

    train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)

    if i % log_steps == 0:
        policy_loss, value_loss, total_loss = train_stats["ppo/loss/policy"], train_stats["ppo/loss/value"], train_stats["ppo/loss/total"]
        print(f"Policy loss: {policy_loss}, Value loss: {value_loss}, Total loss: {total_loss}")
        #send to wandb if cba