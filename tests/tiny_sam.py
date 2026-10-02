"""Build a tiny random-weight SAM model + processor on disk (no download).

Only checks that scripts/sam_box_to_mask.py runs; masks from it are random.
"""
from pathlib import Path


def build_tiny_sam(path):
    import torch
    from transformers import SamConfig, SamImageProcessor, SamModel, SamProcessor

    torch.manual_seed(0)
    cfg = SamConfig(
        vision_config=dict(hidden_size=32, output_channels=32, num_hidden_layers=2, num_attention_heads=2,
                           global_attn_indexes=[1], mlp_dim=64, window_size=7, num_pos_feats=16),
        prompt_encoder_config=dict(hidden_size=32),
        mask_decoder_config=dict(hidden_size=32, mlp_dim=64, num_attention_heads=2, iou_head_hidden_dim=32),
    )
    SamModel(cfg).save_pretrained(str(path))
    SamProcessor(SamImageProcessor()).save_pretrained(str(path))
    return str(Path(path))
