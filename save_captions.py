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

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

WEIGHTS_PATHS = {
    # "coco": "data/coco/coco_prefix_best.pt",
    # "coco": "/data/joonl4/CSE481N/UW-NLP-Capstone-SP22/coco_train/coco_prefix_best.pt"
    "coco": "/data/joonl4/CSE481N/UW-NLP-Capstone-SP22/pretrain/coco_prefix_best.pt"
}

D = torch.device
CPU = torch.device("cuda:0")


class Predictor:
    def setup(self, args):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda")
        self.clip_model, self.preprocess = clip.load(
            "ViT-B/32", device=self.device, jit=False
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.models = {}
        self.prefix_length = 40
        for key, weights_path in WEIGHTS_PATHS.items():
            model = ClipCaptionPrefix(
                self.prefix_length,
                clip_length = args.prefix_length_clip,
                prefix_size= 640 if args.is_rn else 512,
                num_layers = args.num_layers,
                mapping_type=args.mapping_type,
                refinement=False,
                clip_model=self.clip_model,
                tokenizer=GPT2Tokenizer.from_pretrained("gpt2"))
            model.load_state_dict(torch.load(weights_path, map_location=CPU))
            model = model.eval()
            model = model.to(self.device)
            self.models[key] = model

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
            # prefix_embed_1 = model.clip_project(prefix).reshape(1, self.prefix_length, -1)
            # init_caption = generate2(model, self.tokenizer, embed=prefix_embed_1)
            # out = clip.tokenize(init_caption).to(self.device)
            # out = self.clip_model.encode_text(out)
            # prefix_2 = prefix + out
            # prefix_embed = model.clip_project2(prefix_2).reshape(1, self.prefix_length, -1)
            prefix_embed = model.clip_project(prefix).reshape(1, self.prefix_length, -1)
        if use_beam_search:
            return generate_beam(model, self.tokenizer, embed=prefix_embed)[0]
        else:
            return generate2(model, self.tokenizer, embed=prefix_embed)

def map_images_id_to_pathname():
    id_to_pathname = dict()

    # add val data map
    f = json.load(open('data/coco/annotations/captions_val2014.json', 'r'))

    for i in range(len(f['images'])):
        id_to_pathname[str(f['images'][i]['id'])] = ('data/coco/val2014/', f['images'][i]['file_name'])

    # add train data map
    f = json.load(open('data/coco/annotations/captions_train2014.json', 'r'))

    for i in range(len(f['images'])):
        id_to_pathname[str(f['images'][i]['id'])] = ('data/coco/train2014/', f['images'][i]['file_name'])

    print("Len of id_to_pathname is:", len(id_to_pathname))
    torch.save(id_to_pathname, 'data/coco/id_to_pathname.pt')

    return id_to_pathname


def get_karpathy_image_ids(path='data/coco/annotations/val_caption.json'):
    f = json.load(open(path, 'r'))

    result = [element['image_id'] for element in f]
    print("Length of elements in", path, "is", len(result))
    return result


def main(args):
    model = "coco"
    use_beam_search = True
    with open(f'./data/coco/annotations/{args.run_type}_caption.json', 'r') as f:
        data = json.load(f)

    # id_to_pathname = map_images_id_to_pathname()
    # karpathy_val_image_ids = set(get_karpathy_image_ids())

    # print("Len karpathy_val_image_ids is", len(karpathy_val_image_ids))

    predictor = Predictor()
    predictor.setup(args)
    run_type = args.run_type
    out_path = f"./data/coco/oscar_split_clipcap_base_{run_type}.pkl"
    clip_model_name = args.clip_model_type.replace('/', '_')
    clip_model, preprocess = clip.load(args.clip_model_type, device="cuda:0", jit=False)
    all_embeddings = []
    all_captions = []

    # for i, id in enumerate(tqdm(karpathy_val_image_ids)):
    for i in tqdm(range(len(data))):
        # image_dir = id_to_pathname[id][0]
        # image_path = id_to_pathname[id][1]
        d = data[i]
        img_id = d["image_id"]
        if run_type == "train":    
            filename = f"./data/coco/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        elif run_type == "val":
            filename = f"./data/coco/val2014/COCO_val2014_{int(img_id):012d}.jpg"
        else:
            raise IOError('Invalid runtype')
        result = predictor.predict(filename, model, use_beam_search)

        generated_caption = clip.tokenize(result).to('cuda')
        text_prefix = clip_model.encode_text(generated_caption).detach().cpu()
        all_embeddings.append(text_prefix)
        d["clip_embedding"] = i
        all_captions.append(d)
        if i % 100 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)
            print("Step", i, "-- Image", filename, "-- Caption:", result, "-- feature shape:", torch.cat(all_embeddings, dim=0).shape)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    # parser.add_argument('--clip_model_type', type=str)
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--run_type', default="train", choices=("train", "val"))
    args = parser.parse_args()
    main(args)
