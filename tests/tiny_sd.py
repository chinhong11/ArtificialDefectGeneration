"""Build a tiny random-weight SD inpainting pipeline on disk (no download).

Only for testing that the training / generation code runs end-to-end on CPU.
The output images are noise - that's expected.
"""
import json
from pathlib import Path


def bytes_to_unicode():
    """GPT-2 / CLIP byte -> unicode table."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def build_tiny_inpaint_model(path):
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    path = Path(path)
    torch.manual_seed(0)
    tok_dir = path / "_tok"
    tok_dir.mkdir(parents=True, exist_ok=True)
    chars = list(bytes_to_unicode().values())
    vocab = {c: i for i, c in enumerate(chars + [c + "</w>" for c in chars])}
    vocab["<|startoftext|>"] = len(vocab)
    vocab["<|endoftext|>"] = len(vocab)
    (tok_dir / "vocab.json").write_text(json.dumps(vocab))
    (tok_dir / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"), model_max_length=77)

    text_encoder = CLIPTextModel(CLIPTextConfig(
        bos_token_id=vocab["<|startoftext|>"], eos_token_id=vocab["<|endoftext|>"], pad_token_id=1,
        hidden_size=32, intermediate_size=37, num_attention_heads=4, num_hidden_layers=2,
        vocab_size=len(vocab), max_position_embeddings=77))
    unet = UNet2DConditionModel(
        block_out_channels=(32, 64), layers_per_block=1, sample_size=32, in_channels=9, out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), cross_attention_dim=32)
    vae = AutoencoderKL(
        block_out_channels=[32, 64], in_channels=3, out_channels=3, latent_channels=4,
        down_block_types=["DownEncoderBlock2D"] * 2, up_block_types=["UpDecoderBlock2D"] * 2)
    pipe = StableDiffusionInpaintPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=DDPMScheduler(),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.save_pretrained(str(path))
    return str(path)
