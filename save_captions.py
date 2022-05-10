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

# Used to save text feature embedding from first iteration of the clipcap model.

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


# D = torch.device
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

    def predict(self, image, model, use_beam_search):
        """Run a single prediction on the model"""
        image = io.imread(image)
        model = self.models[model]
        pil_image = PIL.Image.fromarray(image)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image).to(
                self.device, dtype=torch.float32
            )
            prefix_embed = model.clip_project(prefix).reshape(1, self.prefix_length, -1)
        if use_beam_search:
            return generate_beam(model, self.tokenizer, embed=prefix_embed)[0]
        else:
            return generate2(model, self.tokenizer, embed=prefix_embed)
    
    def predict2(self, sample, model, use_beam_search):
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
    model = "coco"
    use_beam_search = True
    run_type = args.run_type
    tag = args.tag
    out_path = f"./data/coco/oscar_split_{tag}_{run_type}.pkl"
    device = torch.device("cuda:0")
    clip_model, preprocess = clip.load(args.clip_model_type, device=device, jit=False)
    clip_model.eval()
    predictor = Predictor()
    predictor.setup(args)

    dataset = ClipCocoDataset(args.data, args.prefix_length, 
        normalize_prefix=args.normalize_prefix,
        text_data_path=args.text_data,
        )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    print(f"Validation dataset size is {len(dataloader)}")
    all_embeddings = []
    all_captions = []
    for i, (tokens, mask, prefix) in enumerate(tqdm(dataloader)):
        tokens, mask, prefix = tokens.to(DEVICE), mask.to(DEVICE), prefix.to(DEVICE, dtype=torch.float32)
        sample = (tokens, mask, prefix)
        result = predictor.predict2(sample, model, use_beam_search)
        generated_caption = clip.tokenize(result).to(device)
        text_prefix = clip_model.encode_text(generated_caption).detach().cpu()
        all_embeddings.append(text_prefix)
        all_captions.append(dataset.captions[i])
        if i % 100 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)
            print("Step", i, "-- Image", dataset.image_ids[i], "-- Caption:", result, "-- feature shape:", torch.cat(all_embeddings, dim=0).shape)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/coco/oscar_split_train.pkl')
    parser.add_argument('--text_data', default=None)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--run_type', default="train", choices=("train", "val"))
    parser.add_argument('--weights', type=str, required=True)
    args = parser.parse_args()
    main(args)
