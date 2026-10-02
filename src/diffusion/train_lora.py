"""DreamBooth-style LoRA fine-tuning of a Stable Diffusion INPAINTING model on
defect crops, so it learns to paint "your" defect inside a mask.

The inpainting UNet sees  [noisy latent (4) | mask (1) | masked-image latent (4)].
We train with mask = (dilated) real defect mask, so the model learns:
"given this surrounding surface and this region, paint a <defect> here".

Example (one defect type, 12 GB GPU):
    accelerate launch -m src.diffusion.train_lora \
        --data data/metal_part/crops/scratch \
        --output outputs/lora_scratch \
        --val-good-dir data/metal_part/good \
        --max-steps 2000 --rank 8 --lr 1e-4 \
        --mixed-precision fp16 --gradient-checkpointing

(`python -m src.diffusion.train_lora ...` also works on a single GPU.)

Watch <output>/samples/step_*.png and pick the checkpoint that looks real but
does NOT just copy training defects (that is overfitting).
"""
import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.common import dilate, list_images, read_mask, read_rgb


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Folder from prepare_crops (metadata.jsonl, images/, masks/)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-model", default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    ap.add_argument("--caption", help="Override caption from metadata.jsonl")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--rank", type=int, default=8, help="LoRA rank (4-16 is typical)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=2000, help="Optimizer steps")
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--mask-dilate", type=int, default=8, help="Grow defect mask by N px (at crop resolution)")
    ap.add_argument("--mask-dilate-jitter", type=int, default=8, help="Extra random dilation 0..N px")
    ap.add_argument("--mask-loss-weight", type=float, default=2.0,
                    help="Loss weight inside the mask (1 = uniform). >1 focuses learning on the defect")
    ap.add_argument("--mixed-precision", default="no", choices=["no", "fp16", "bf16"])
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--use-8bit-adam", action="store_true")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--sample-every", type=int, default=250)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--val-good-dir", help="Good images used for preview samples (random crops)")
    ap.add_argument("--sample-steps", type=int, default=30)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


class DefectCropDataset(Dataset):
    def __init__(self, root, resolution, caption=None, dilate_px=8, dilate_jitter=8):
        self.root = Path(root)
        self.items = [json.loads(l) for l in (self.root / "metadata.jsonl").read_text().splitlines() if l.strip()]
        if not self.items:
            raise ValueError(f"No entries in {self.root / 'metadata.jsonl'}")
        self.res, self.caption = resolution, caption
        self.dilate_px, self.dilate_jitter = dilate_px, dilate_jitter

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = read_rgb(self.root / it["file_name"])
        mask = read_mask(self.root / it["mask"])
        if img.shape[0] != self.res or img.shape[1] != self.res:
            img = cv2.resize(img, (self.res, self.res), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.res, self.res), interpolation=cv2.INTER_NEAREST)
        # same random flips / 90-degree rotation for image and mask
        if random.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            img, mask = img[::-1], mask[::-1]
        k = random.randint(0, 3)
        img, mask = np.rot90(img, k), np.rot90(mask, k)
        mask = dilate(np.ascontiguousarray(mask), self.dilate_px + random.randint(0, self.dilate_jitter))

        pixel = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        m = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        masked = pixel * (m < 0.5)  # same convention as StableDiffusionInpaintPipeline
        return {"pixel_values": pixel, "mask": m, "masked_image": masked,
                "caption": self.caption or it["caption"]}


def random_good_crop(good_paths, size, rng):
    img = read_rgb(rng.choice(good_paths))
    h, w = img.shape[:2]
    side = min(size, h, w)
    y0, x0 = rng.randint(0, h - side), rng.randint(0, w - side)
    c = img[y0:y0 + side, x0:x0 + side]
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA) if side != size else c


@torch.no_grad()
def save_samples(pipe, dataset, good_paths, args, step, device):
    """Preview: paint the defect into good-image crops using training mask shapes."""
    from PIL import Image

    rng = random.Random(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    tiles = []
    for i in range(args.num_samples):
        it = dataset.items[i % len(dataset.items)]
        mask = dilate(read_mask(dataset.root / it["mask"]), args.mask_dilate)
        mask = cv2.resize(mask, (args.resolution, args.resolution), interpolation=cv2.INTER_NEAREST)
        base = (random_good_crop(good_paths, args.resolution, rng) if good_paths
                else read_rgb(dataset.root / it["file_name"]))
        base = cv2.resize(base, (args.resolution, args.resolution))
        out = pipe(prompt=args.caption or it["caption"], image=Image.fromarray(base),
                   mask_image=Image.fromarray(mask), height=args.resolution, width=args.resolution,
                   num_inference_steps=args.sample_steps, generator=gen).images[0]
        overlay = base.copy()
        cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (255, 0, 0), 2)
        tiles.append(np.concatenate([overlay, np.asarray(out)], axis=1))
    grid = np.concatenate(tiles, axis=0)
    path = Path(args.output) / "samples" / f"step_{step:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(path)
    print(f"  samples -> {path}  (left: good crop + mask, right: generated)")


def save_lora(unet, out_dir):
    from diffusers import StableDiffusionInpaintPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionInpaintPipeline.save_lora_weights(save_directory=str(out_dir), unet_lora_layers=state,
                                                     safe_serialization=True)
    print(f"  LoRA saved -> {out_dir}")


def main(argv=None):
    args = parse_args(argv)
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig
    from transformers import CLIPTextModel, CLIPTokenizer

    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.grad_accum)
    set_seed(args.seed)
    random.seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=2))

    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    if unet.config.in_channels != 9:
        raise ValueError(f"{args.base_model} is not an inpainting model (UNet in_channels="
                         f"{unet.config.in_channels}, expected 9)")

    for m in (vae, text_encoder, unet):
        m.requires_grad_(False)
    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.mixed_precision, torch.float32)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    for p in lora_params:  # trainable weights stay fp32 for stable mixed-precision training
        p.data = p.data.float()
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    print(f"Trainable LoRA params: {sum(p.numel() for p in lora_params) / 1e6:.2f}M")

    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(lora_params, lr=args.lr, weight_decay=1e-2)
    else:
        optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-2)

    dataset = DefectCropDataset(args.data, args.resolution, args.caption, args.mask_dilate, args.mask_dilate_jitter)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        drop_last=len(dataset) >= args.batch_size)
    lr_sched = get_scheduler("constant_with_warmup", optimizer, num_warmup_steps=args.warmup_steps,
                             num_training_steps=args.max_steps)
    unet, optimizer, loader, lr_sched = accelerator.prepare(unet, optimizer, loader, lr_sched)
    good_paths = list_images(args.val_good_dir) if args.val_good_dir else []

    def encode_text(captions):
        ids = tokenizer(captions, padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt").input_ids.to(accelerator.device)
        return text_encoder(ids)[0]

    def make_pipe():
        return StableDiffusionInpaintPipeline.from_pretrained(
            args.base_model, unet=accelerator.unwrap_model(unet), vae=vae, text_encoder=text_encoder,
            tokenizer=tokenizer, safety_checker=None, requires_safety_checker=False, torch_dtype=weight_dtype,
        ).to(accelerator.device)

    print(f"{len(dataset)} crops, {args.max_steps} steps, effective batch {args.batch_size * args.grad_accum}")
    step, epochs = 0, math.ceil(args.max_steps * args.grad_accum / max(1, len(loader)))
    unet.train()
    for _ in range(epochs):
        for batch in loader:
            with accelerator.accumulate(unet):
                pixel = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                masked = batch["masked_image"].to(accelerator.device, dtype=weight_dtype)
                mask = batch["mask"].to(accelerator.device, dtype=weight_dtype)
                with torch.no_grad():
                    sf = vae.config.scaling_factor
                    latents = vae.encode(pixel).latent_dist.sample() * sf
                    masked_latents = vae.encode(masked).latent_dist.sample() * sf
                    text = encode_text(batch["caption"])
                mask_lat = F.interpolate(mask, size=latents.shape[-2:], mode="nearest")

                noise = torch.randn_like(latents)
                t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                  device=latents.device).long()
                noisy = noise_scheduler.add_noise(latents, noise, t)
                model_in = torch.cat([noisy, mask_lat, masked_latents], dim=1)
                pred = unet(model_in, t, text).sample

                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, t)
                else:
                    target = noise
                weight = 1.0 + (args.mask_loss_weight - 1.0) * mask_lat
                loss = (F.mse_loss(pred.float(), target.float(), reduction="none") * weight.float()).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue
            step += 1
            if step % 10 == 0 or step == 1:
                print(f"step {step}/{args.max_steps}  loss {loss.item():.4f}")
            if accelerator.is_main_process:
                if step % args.save_every == 0:
                    save_lora(accelerator.unwrap_model(unet), out / f"checkpoint-{step}")
                if args.sample_every and step % args.sample_every == 0:
                    unet.eval()
                    save_samples(make_pipe(), dataset, good_paths, args, step, accelerator.device)
                    unet.train()
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    if accelerator.is_main_process:
        save_lora(accelerator.unwrap_model(unet), out)
        if args.sample_every:
            unet.eval()
            save_samples(make_pipe(), dataset, good_paths, args, step, accelerator.device)
        if torch.cuda.is_available():
            print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    accelerator.end_training()


if __name__ == "__main__":
    main()
