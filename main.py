import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "CapSpeech"))

import math
import torch
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from daam.utils import pick_inference_device
from capspeech.nar.generate import load_model, encode, get_duration, seed_everything
from capspeech.nar.utils import make_pad_mask
from daam_capspeech import extract_attn, aggregate_mean_attn, visualize_maps

EXAMPLES = [
    {
        "transcript": "Hello world.",
        "caption": "A calm male voice.",
    },
    {
        "transcript": "The weather is so nice today because of the bright yellow sun.",
        "caption": "A cheerful female voice that peaks in the beginning and then fades out.",
    },
    {
        "transcript": "I love music and I love dance.",
        "caption": "A warm, expressive voice that peaks in the middle and then fades out.",
    },
    {
        "transcript": "Good morning everyone, I am happy to be here today.",
        "caption": "A deep, slow male voice that gets more excited towards the end.",
    },
    {
        "transcript": "Thank you very much for your time and your attention.",
        "caption": "A soft, gentle female voice that starts off quiet, screams in the middle, then gets quieter towards the end.",
    },
]


def main():
    # config variables
    task = config.TASK
    output_dir = config.OUTPUT_DIR
    device = config.DEVICE
    steps = config.STEPS
    cfg = config.CFG
    max_tokens = config.MAX_TOKENS
    seed = config.SEED

    if device == "auto":
        device = pick_inference_device()

    seed_everything(seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Device: {device}")
    model_list = load_model(device, task)
    (model, vocoder, phn2num, text_tokenizer, clap_model,
     duration_tokenizer, duration_model, caption_tokenizer, caption_encoder) = model_list

    # iterate over examples
    for idx, example in enumerate(EXAMPLES, start=1):
        transcript = example["transcript"]
        caption = example["caption"]
        print(f"\n{'='*60}")
        print(f"Example {idx}: {transcript[:60]}...")

        tag = "none"
        # encode transcript into phoneme tokens
        phn = encode(transcript, text_tokenizer)
        text_tokens = [phn2num[p] for p in phn]
        text = torch.LongTensor(text_tokens).unsqueeze(0).to(device)

        with torch.no_grad():
            # tokenize caption and encode it into a tensor
            batch_enc = caption_tokenizer(caption, return_tensors="pt")
            ori_token_ids = batch_enc["input_ids"].to(device)
            prompt = caption_encoder(input_ids=ori_token_ids).last_hidden_state.squeeze().unsqueeze(0).to(device)

            tag_embed = clap_model.get_text_embedding([tag], use_tensor=True)
            clap = tag_embed.squeeze().unsqueeze(0).to(device)

            duration_inputs = caption + " <NEW_SEP> " + transcript
            duration_inputs = duration_tokenizer(
                duration_inputs, return_tensors="pt",
                padding="max_length", truncation=True, max_length=400,
            )
            predicted_dur = duration_model(**duration_inputs).logits.squeeze().item()
            duration = get_duration(transcript, predicted_dur)

        audio_clips = torch.zeros([1, math.ceil(duration * 24000 / 256), 100]).to(device)
        seq_len_prompt = prompt.shape[1]
        prompt_lens = torch.Tensor([seq_len_prompt])
        prompt_mask = make_pad_mask(prompt_lens, seq_len_prompt).to(prompt.device)

        result = extract_attn(
            model, vocoder, audio_clips, None, text, prompt, clap, prompt_mask,
            steps=steps, cfg=cfg, sway_sampling_coef=-1.0, device=device,
        )

        mel_spec = result["mel"]
        n_mels = mel_spec.shape[1]
        T_spec = mel_spec.shape[2]

        heatmaps = aggregate_mean_attn(
            result["attention_mean"], n_mels=n_mels, T_spec=T_spec,
        )

        token_ids = ori_token_ids.squeeze().tolist()
        token_labels = caption_tokenizer.convert_ids_to_tokens(token_ids)

        fig_path = os.path.join(output_dir, f"example_{idx}.png")
        wav_path = os.path.join(output_dir, f"example_{idx}.wav")

        visualize_maps(
            heatmaps, mel_spec, token_labels,
            save_path=fig_path, max_tokens=max_tokens,
        )
        plt.close("all")

        sf.write(wav_path, result["wav"], 24000)
        print(f"  Saved: {fig_path}, {wav_path}")

    print(f"\nAll 5 examples saved to {output_dir}/")


if __name__ == "__main__":
    main()
