import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint

import model
import lightning_wrapper
import data_util
import text_dataset


if __name__ == "__main__":
    sequence_length = 64
    embed_dim = 100

    train_path = Path("train.txt")
    glove_path = Path("glove.txt")
    words, embeddings = data_util.load_glove(glove_path, dim=embed_dim)
    indexer = data_util.SymbolIndexer(words)

    dataset = text_dataset.TextDataset(sequence_length, train_path, indexer=indexer)
    function = model.BasicModel(sequence_length, embeddings)

    loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=6, shuffle=True)
    checkpoint_callback = ModelCheckpoint(every_n_train_steps=1024)
    trainer = pl.Trainer(gpus=0, callbacks=[checkpoint_callback])

    trainer.fit(lightning_wrapper.LightningWrapper(function), loader)
