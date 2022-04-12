import torch
import pytorch_lightning
from pathlib import Path

import data_util
import text_dataset
import lightning_wrapper
import model


if True:
    pytorch_lightning.seed_everything(0)

    target_count = 8192

    k = 3
    sequence_length = 64
    embed_dim = 192

    indexer = data_util.SymbolIndexer()

    function = model.BasicModel(sequence_length, indexer, embed_dim)
    function = lightning_wrapper.LightningWrapper.load_from_checkpoint(input("path: "), map_location="cpu", f=function).f

    text = text_dataset.TextDataset(sequence_length, Path("data") / Path("test.txt"))

    loader = torch.utils.data.DataLoader(text, batch_size=1, num_workers=6, shuffle=True)


    correct = 0
    total = 0

    for sample in text:
        sample = sample.squeeze(0)
        if total >= target_count:
           break        

        pred = function(sample.unsqueeze(0)).squeeze(0)[-2]
        pred = function.embed.interpret(pred, k=k)
        expected = indexer.to_symbol(sample[-1].item())

        # print("" .join([indexer.to_symbol(index.item()) for index in sample]))
        # print(pred)
        # print(expected)

        total += 1
        correct += (expected in pred)

    print (correct / total)
