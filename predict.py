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
    "coco": "/data/joonl4/CSE481N/UW-NLP-Capstone-SP22/coco_train/coco_prefix_best.pt"
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
            model = ClipCaptionModel(
                self.prefix_length,
                clip_length = args.prefix_length_clip,
                prefix_size= 640 if args.is_rn else 512,
                num_layers = args.num_layers,
                mapping_type=args.mapping_type,
                refinement=True,
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
            prefix_embed_1 = model.clip_project(prefix).reshape(1, self.prefix_length, -1)
            # generate refinedment prediction
            init_caption = generate2(model, self.tokenizer, embed=prefix_embed_1)
            out = clip.tokenize(init_caption).to(self.device)
            out = model.clip_model.encode_text(out)
            prefix_2 = prefix + out
            prefix_embed = model.clip_project2(prefix_2).reshape(1, self.prefix_length, -1)
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
    id_to_pathname = map_images_id_to_pathname()
    karpathy_val_image_ids = set(get_karpathy_image_ids())

    print("Len karpathy_val_image_ids is", len(karpathy_val_image_ids))

    val_pred_captions = list()
    model = "coco"
    use_beam_search = True

    predictor = Predictor()
    predictor.setup(args)

    for i, id in enumerate(tqdm(karpathy_val_image_ids)):
        image_dir = id_to_pathname[id][0]
        image_path = id_to_pathname[id][1]

        result = predictor.predict(image_dir + image_path, model, use_beam_search)
        val_pred_captions.append({"image_id" : id, "caption" : result})

        if i % 100 == 0:
            json.dump(val_pred_captions, open("data/coco/annotations/pred_val_caption.json", "w"))
            print("Step", i, "-- Image", image_path, "-- Caption:", result)

    json.dump(val_pred_captions, open("data/coco/annotations/pred_val_caption.json", "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--clip_model_type', type=str)
    args = parser.parse_args()
    main(args)
