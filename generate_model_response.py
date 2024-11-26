import time
import torch


def generate_model_response(text, emotion, model, tokenizer, max_length=100):
    start_time = time.time()

    with torch.no_grad():
        prompt = f'''<{emotion}> {text} </{emotion}>'''
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260, 128261]], dtype=torch.int64)

        # Concatenate the tensors
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

        input_ids = modified_input_ids  # Ensure input IDs are on the correct GPU
        input_ids=  input_ids.to('cuda')

        generation_kwargs = {
            "pad_token_id": 128258,
            "top_k": 50,
            "temperature": 0.5,
            "repetition_penalty": 1.1,
            "max_length": max_length,
            "eos_token_id": 128258
        }

        response_tensor = model.generate(input_ids, **generation_kwargs)

        return input_ids, response_tensor
