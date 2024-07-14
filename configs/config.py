from email.policy import default
import os

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2  # type: ignore

import argparse
import json
import os
from typing import Any, Dict, List

parser = argparse.ArgumentParser(
    description=(
        "Ref-AVS, ECCV'2024."
    )
)

parser.add_argument(
    "--train_params",
    type=list,
    default=[
        'audio_proj',
        'text_proj',
        'prompt_proj',
        'avs_adapt',
        'ref_avs_attn',
    ],
    help="Text model to extract textual reference feature.",
)

parser.add_argument(
    "--text_model",
    type=str,
    default='distilbert/distilroberta-base',
    help="Text model to extract textual reference feature.",
)

parser.add_argument(
    "--exp",
    type=str,
    required=True,
    help="exp to run.",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="The path to load the refavs model checkpoints."
)
parser.add_argument("--save_ckpt", type=str, default='./ckpt', help='Checkpoints save dir.')
parser.add_argument("--log_path", type=str, default='./logs', help='Log info save path.')

file_arch = """
./data/REFAVS
    - /media
    - /gt_mask
    - /metadata.csv
"""
print(f">>> File arch: {file_arch}")
parser.add_argument(
    "--data_dir",
    type=str,
    default='./data/REFAVS',
    help=f"The data paranet dir. File arch should be: {file_arch}"
)

parser.add_argument("--show_params", action='store_true', help=f"Show params names with Requires_grad==True.")
parser.add_argument("--m2f_model", type=str, default='facebook/mask2former-swin-base-ade-semantic', help="Pretrained mask2former.")

parser.add_argument("--lr", type=float, default=1e-4, help='lr to fine tuning adapters.')
parser.add_argument("--epochs", type=int, default=50, help='epochs to fine tuning adapters.')
parser.add_argument("--loss", type=str, default='bce', help='')

parser.add_argument("--train", default=False, action='store_true', help='start train?')
parser.add_argument("--val", type=str, default=None, help='type: str; val | test')  # NOTE: for test and val.
parser.add_argument("--test", default=False, action='store_true', help='start test?')


parser.add_argument("--gpu_id", type=str, default="0", help="The GPU device to run generation on.")

parser.add_argument("--run", type=str, default='train', help="train, test")

parser.add_argument("--frame_n", type=int, default=10, help="Frame num of each video. Fixed to 10.")
parser.add_argument("--text_max_len", type=int, default=25, help="Maximum textual reference length.")



args = parser.parse_args()

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
print(f'>>> Sys: set "CUDA_VISIBLE_DEVICES" - GPU: {args.gpu_id}')
