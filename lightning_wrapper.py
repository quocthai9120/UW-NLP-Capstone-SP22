from numpy import diagonal
import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl


def _barlow_twins_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    LAMBDA = 0.005
    a = a.mean(dim=1)
    b = b.mean(dim=1)

    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)

    C = (a.transpose(0, 1) @ b)
    diag = C.diagonal()
    similarity = (diag - 1.0).pow(2).sum()
    redundancy = (C - diag.diag()).pow(2).sum()

    return similarity + LAMBDA * redundancy

class LightningWrapper(pl.LightningModule):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)

    def training_step(self, batch: torch.Tensor, _):
        x0, x1 = batch
        y0, y1 = self(x0), self(x1)

        loss = _barlow_twins_loss(y0, y1)

        self.log("train_loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.parameters(), lr=1.0)
        return optimizer