import pytorch_lightning as pl
import torch

class InferenceCustomIsIcModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        out = torch.nn.functional.sigmoid(out)
        return out
