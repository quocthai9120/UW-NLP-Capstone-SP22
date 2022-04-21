import torch
import skimage.io as io
import ClIP.clip as clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse


def main(clip_model_type: str, run_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/coco/oscar_split_{clip_model_name}_{run_type}.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open(f'./data/coco/annotations/{run_type}_caption.json', 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_embedding_sequences = []
    all_captions = []
    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["image_id"]
        if run_type == "train":    
            filename = f"./data/coco/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        elif run_type == "val":
            filename = f"./data/coco/val2014/COCO_val2014_{int(img_id):012d}.jpg"
        else:
            raise IOError('Invalid runtype')
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix, prefix_sequence = clip_model.encode_image_with_sequence_embedding(image)
        d["clip_embedding"] = i
        all_embeddings.append(prefix.cpu())
        all_embedding_sequences.append(prefix_sequence.cpu())
        all_captions.append(d)

        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({
                    "clip_embedding": torch.cat(all_embeddings, dim=0),
                    "clip_embedding_sequences": torch.cat(all_embedding_sequences, dim=0),
                    "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
            pickle.dump({
                "clip_embedding": torch.cat(all_embeddings, dim=0),
                "clip_embedding_sequences": torch.cat(all_embedding_sequences, dim=0),
                "captions": all_captions}, f)
    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--run_type', default="train", choices=("train", "val"))
    args = parser.parse_args()
    exit(main(args.clip_model_type, args.run_type))
