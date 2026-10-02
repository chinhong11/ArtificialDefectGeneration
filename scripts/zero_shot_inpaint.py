"""Step 2 sanity test: pretrained SD inpainting, NO fine-tuning.

Checks that the whole pipeline (window, mask, inpaint, paste-back, labels)
works and shows how unrealistic the defects are before training a LoRA.

Example:
    python scripts/zero_shot_inpaint.py --good-dir data/metal_part/good \
        --prompt "a thin scratch on a metal surface" --out outputs/zero_shot --num 10

Accepts every option of src/diffusion/inpaint_generate.py (except --lora).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.diffusion.inpaint_generate import main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--lora" in argv:
        sys.exit("zero_shot_inpaint.py runs without LoRA; use src/diffusion/inpaint_generate.py instead")
    if "--num" not in argv:
        argv += ["--num", "10"]
    if "--mask-mode" not in argv and "--real-mask-dir" not in argv:
        argv += ["--mask-mode", "mixed"]
    main(argv)
