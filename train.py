from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import clip
from network import *
from loss_manager import LossManager

class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix

    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False, text_data_path=None):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))



def train(train_dataset: ClipCocoDataset, val_dataset: ClipCocoDataset, model: ClipCaptionModel, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):

    writer = SummaryWriter(log_dir=f"./logs/{output_prefix}")
    device = torch.device('cuda:0')
    batch_size = args.bs
    lm = LossManager()
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    # save_config(args)
    best_val = float('inf')
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        model.train()
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            loss_dict = lm.get_loss(outputs, tokens)
            loss = loss_dict["loss/total"]
            loss.backward()
            print(module_grad_stats(model))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"train_loss": loss.item()})
            progress.update()
            for k, v in loss_dict.items():
                writer.add_scalar(
                    f"Train/{k}",
                    float(v),
                    len(train_dataloader) * epoch + idx
                )
        progress.close()
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, f"{output_prefix}_{epoch}.pt"),
        )
        model.eval()
        losses = []
        progress = tqdm(total=len(val_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix) in enumerate(val_dataloader):
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            loss_dict = lm.get_loss(outputs, tokens)
            loss = loss_dict["loss/total"]
            progress.set_postfix({"val_loss": loss.item()})
            progress.update()
            losses.append(float(loss))
            for k, v in loss_dict.items():
                writer.add_scalar(
                    f"Val/{k}",
                    float(v),
                    len(val_dataloader) * epoch + idx
                )
        progress.close()
        val = np.mean(losses)
        writer.add_scalar(
            'Val/epochloss',
            val,
            len(val_dataloader) * epoch + idx)
        if val < best_val:
            val = best_val
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}_best.pt"),
            )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/coco/oscar_split_train.pkl')
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=40)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--clip_model_type', type=str)
    args = parser.parse_args()
    prefix_length = args.prefix_length
    device = torch.device('cuda:0')
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    train_dataset = ClipCocoDataset(args.data, prefix_length, normalize_prefix=args.normalize_prefix)
    val_dataset = ClipCocoDataset(args.data.replace("train", "val"), prefix_length, normalize_prefix=args.normalize_prefix)
    prefix_dim = 640 if args.is_rn else 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type, refinement=True, clip_model=clip_model, tokenizer=GPT2Tokenizer.from_pretrained("gpt2"))
        # arbitrary finetuning
        model.load_state_dict(torch.load("/data/joonl4/CSE481N/UW-NLP-Capstone-SP22/pretrain/coco_prefix_best.pt"), strict=False)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train both prefix and GPT")
        sys.stdout.flush()
    train(train_dataset, val_dataset, model, args, output_dir=args.out_dir, output_prefix=args.prefix)


if __name__ == '__main__':
    main()
