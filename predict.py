import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
import skimage.io as io
import PIL.Image
from train import TransformerMapper
from tqdm import tqdm

from os import listdir
from os.path import isfile, join
import json
from train import *
from utils import *


# N = type(None)
# V = np.array
# ARRAY = np.ndarray
# ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
# VS = Union[Tuple[V, ...], List[V]]
# VN = Union[V, N]
# VNS = Union[VS, N]
# T = torch.Tensor
# TS = Union[Tuple[T, ...], List[T]]
# TN = Optional[T]
# TNS = Union[Tuple[TN, ...], List[TN]]
# TSN = Optional[TS]
# TA = Union[T, ARRAY]

# WEIGHTS_PATHS = {
#     # "coco": "data/coco/coco_prefix_best.pt",
#     "coco": "/data/joonl4/CSE481N/UW-NLP-Capstone-SP22/pretrain/coco_prefix_best.pt"
#     # "coco": "/data/joonl4/CSE481N/UW-NLP-Capstone-SP22/coco_train/coco_prefix_best.pt"
#     # refinement
#     # "coco": "/data/joonl4/CSE481N/UW-NLP-Capstone-SP22/refinement_v1/coco-prefix_refinment-v1_best.pt",
#     # "coco": "/data/joonl4/CSE481N/UW-NLP-Capstone-SP22/refinement_v1-concat/coco-prefix_refinment-v1-concat_6.pt",
#     # "coco": "/data/joonl4/CSE481N/UW-NLP-Capstone-SP22/refinement_v1-concat/coco-prefix_refinment-v1-concat_best.pt",
#     # "base_model": "/data/joonl4/CSE481N/UW-NLP-Capstone-SP22/pretrain/coco_prefix_best.pt"
# }

DEVICE = torch.device("cuda:0")


class Predictor:
    def setup(self, args):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda")
        self.clip_model, self.preprocess = clip.load(
            "ViT-B/32", device=self.device, jit=False
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # self.models = {}
        self.prefix_length = args.prefix_length
        # for key, weights_path in WEIGHTS_PATHS.items():
        weights_path = args.weights
        prefix_size= 640 if args.is_rn else 512
        if args.text_data is not None:
            prefix_size *= 2

        if args.only_prefix:
            model = ClipCaptionPrefix(
                self.prefix_length,
                clip_length=args.prefix_length_clip,
                prefix_size=prefix_size,
                num_layers=args.num_layers,
                mapping_type=args.mapping_type,
            )
        else:
            model = ClipCaptionModel(
                self.prefix_length,
                clip_length = args.prefix_length_clip,
                prefix_size=prefix_size,
                num_layers = args.num_layers,
                mapping_type=args.mapping_type,
            )
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model = model.eval()
        model = model.to(self.device)
        self.model = model

    def predict(self, sample, use_beam_search, base_model="base_model"):
        """Run a single prediction on the model"""
        (tokens, mask, prefix) = sample
        with torch.no_grad():
            prefix_embed = self.model(tokens, prefix, mask)["prefix"]
            # prefix_embed = self.model.clip_project(prefix).reshape(1, self.prefix_length, -1)
        if use_beam_search:
            return generate_beam(self.model, self.tokenizer, embed=prefix_embed)[0]
        else:
            return generate2(self.model, self.tokenizer, embed=prefix_embed)


def main(args):
    val_dataset = ClipCocoDataset(args.data, args.prefix_length, 
        normalize_prefix=args.normalize_prefix,
        text_data_path=args.text_data,
        unique=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)
    print(f"Validation dataset size is {len(val_dataloader)}")

    val_pred_captions = list()
    model = "coco"
    use_beam_search = True
    save_tag = args.tag

    predictor = Predictor()
    predictor.setup(args)
    ids = val_dataset.image_ids
    assert len(ids) == len(val_dataloader), "number of image_ids does not match dataloader size!"
    for i, (tokens, mask, prefix) in enumerate(tqdm(val_dataloader)):
        tokens, mask, prefix = tokens.to(DEVICE), mask.to(DEVICE), prefix.to(DEVICE, dtype=torch.float32)
        result = predictor.predict((tokens, mask, prefix), use_beam_search)
        val_pred_captions.append({"image_id" : ids[i], "caption" : result})

        if i % 100 == 0:
            json.dump(val_pred_captions, open(f"{args.out_dir}/pred_val_caption_{save_tag}.json", "w"))
            print("Step", i, "-- Image_id", ids[i], "-- Caption:", result)

    json.dump(val_pred_captions, open(f"{args.out_dir}/pred_val_caption_{save_tag}.json", "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/coco/oscar_split_train.pkl')
    parser.add_argument('--text_data', default=None)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--clip_model_type', type=str)
    parser.add_argument('--tag', type=str)
    parser.add_argument('--out_dir', default='./checkpoints')
    args = parser.parse_args()
    main(args)
