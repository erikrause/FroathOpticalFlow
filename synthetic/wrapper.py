import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from models.unet import createUnet, get_preprocessing_fn
import torch
from torch.nn import BCEWithLogitsLoss, Sigmoid


class OreSegmentor(pl.LightningModule):
    def __init__(self,
                 learning_rate,
                 weight_decay,
                 *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self._lr = learning_rate
        self._weight_decay = weight_decay
        self._model = createUnet(inputchannels=3, outputchannels=1, is_train=True)

        self._loss = BCEWithLogitsLoss()
        self._sigmoid = Sigmoid()

    def forward(self, colors):
        return self._model(colors)

    def training_step(self, batch, batch_idx):
        colors, segments = batch
        pred = self.forward(colors)
        loss = self._loss(pred, segments)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        colors, segments = batch
        pred = self.forward(colors)
        loss = self._loss(pred, segments) #torch.nn.F.cross_entropy(pred, segments)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        pred_segments = self._sigmoid(pred)     # TODO: add metrics below
        return loss

    def test_step(self, batch, batch_idx):
        colors, segments = batch
        pred = self.forward(colors)
        loss = self._loss(pred, segments)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        pred_segments = self._sigmoid(pred)     # TODO: add metrics below
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self._model.parameters(),
                               lr=self._lr,
                               weight_decay=self._weight_decay)
