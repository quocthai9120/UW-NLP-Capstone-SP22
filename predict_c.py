import clip
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
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
from utils import *
from network_c import ClipCaptionPrefix, ClipCocoDataset
import argparse


DEVICE = torch.device("cuda:0")


class Predictor:
    def setup(self, args):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda")
        self.clip_model, self.preprocess = clip.load(
            args.clip_model_type, device=self.device, jit=False
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
                # mapping_type=args.mapping_type,
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
        self.guided = args.guided

    def predict(self, sample, use_beam_search, base_model="base_model"):
        """Run a single prediction on the model"""
        (tokens, mask, prefix, prefix_sequence) = sample
        with torch.no_grad():
            prefix_embed = self.model(tokens, prefix, prefix_sequence, mask)["prefix"]
            # prefix_embed = self.model.clip_project(prefix).reshape(1, self.prefix_length, -1)
        if use_beam_search:
            return generate_beam(
                self.model,
                self.tokenizer,
                beam_size=5,
                embed=prefix_embed,
                guided=self.guided,
                clip_model=self.clip_model,
                clip_embed=prefix.squeeze(0))[0]
        else:
            return generate2(self.model, self.tokenizer, embed=prefix_embed)


def main(args):
    val_dataset = ClipCocoDataset("val", args.prefix_length, 
        normalize_prefix=args.normalize_prefix,
        # text_data_path=args.text_data,
        unique=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)
    print(f"Validation dataset size is {len(val_dataloader)}")

    val_pred_captions = list()
    model = "coco"
    use_beam_search = args.beam
    print(f"Beam search: {use_beam_search}")
    print(f"Guided: {args.guided}")
    save_tag = args.tag

    predictor = Predictor()
    predictor.setup(args)
    ids = val_dataset.image_ids
    assert len(ids) == len(val_dataloader), "number of image_ids does not match dataloader size!"
    for i, (tokens, mask, prefix, prefix_sequence) in enumerate(tqdm(val_dataloader)):
        tokens, mask, prefix, prefix_sequence = \
            tokens.to(DEVICE), mask.to(DEVICE), prefix.to(DEVICE, dtype=torch.float32), prefix_sequence.to(DEVICE, dtype=torch.float32)
        result = predictor.predict((tokens, mask, prefix, prefix_sequence), use_beam_search)
        val_pred_captions.append({"image_id" : ids[i], "caption" : result})

        if i % 100 == 0:
            json.dump(val_pred_captions, open(f"{args.out_dir}/pred_val_caption_{save_tag}.json", "w"))
            print("Step", i, "-- Image_id", ids[i], "-- Caption:", result)

    json.dump(val_pred_captions, open(f"{args.out_dir}/pred_val_caption_{save_tag}.json", "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/coco/oscar_split_train.pkl')
    parser.add_argument('--text_data', type=str, required=False, default=None)
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
    parser.add_argument('--beam', action='store_true', default=False)
    parser.add_argument('--guided', type=float, default=-1)
    args = parser.parse_args()
    main(args)
