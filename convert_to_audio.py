import torch
import soundfile as sf
from load_facodec import fa_decoder, spk_embs


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



def convert_to_audio(generated_ids, tokenizer):
    
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

