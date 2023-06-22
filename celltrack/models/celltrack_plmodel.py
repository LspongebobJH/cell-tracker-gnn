from typing import Any, List
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import numpy as np

from pytorch_lightning import LightningModule
import celltrack.models.gnn_modules.celltrack_model as celltrack_model
from torchmetrics import Accuracy, Precision, Recall


class CellTrackLitModel(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    """

    def __init__(
        self,
        sample,
        weight_loss,
        directed,
        model_params,
        separate_models,
        loss_weights,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        self.separate_models = separate_models
        if self.separate_models:
            model_attr = getattr(celltrack_model, model_params.target)
            self.model = model_attr(**model_params.kwargs)
        else:
            assert False, "Variable separate_models should be set to True!"
        self.sample = sample
        self.weight_loss = weight_loss

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(loss_weights))

        self.acc, self.prec, self.rec =\
            Accuracy('binary'), Precision('binary'), Recall('binary')

        self.metric_hist = {
            "train/acc": [], "train/prec": [], "train/rec": [], "train/loss": [],
            "val/acc": [], "val/prec": [], "val/rec": [], "val/loss": [],
            'test/acc': [], 'test/prec': [], 'test/rec': [], 'test/loss': []
        }

    def forward(self, x, edge_index, edge_feat):
        return self.model(x, edge_index, edge_feat)

    def _compute_loss(self, outputs, edge_labels):
        edge_sum = edge_labels.sum()
        weight = (edge_labels.shape[0] - edge_sum) / edge_sum if edge_sum else 0.0
        loss = F.binary_cross_entropy_with_logits(outputs.view(-1),
                                                  edge_labels.view(-1),
                                                  pos_weight=weight).to(self.device)
        return loss

    def step(self, batch):
        if self.separate_models:
            x, x_2, edge_index, batch_ind, edge_label, edge_feat = batch.x, batch.x_2, batch.edge_index, batch.batch, batch.edge_label, batch.edge_feat
            y_hat = self.forward((x, x_2), edge_index, edge_feat.float())
        else:
            x, edge_index, batch_ind, edge_label, edge_feat = batch.x, batch.edge_index, batch.batch, batch.edge_label, batch.edge_feat
            y_hat = self.forward(x, edge_index, edge_feat.float())

        loss = self._compute_loss(y_hat, edge_label)
        preds = (y_hat >= 0.5).type(torch.int16)
        edge_label = edge_label.type(torch.int16)
        return loss, preds, edge_label
    
    def logging(self, level, stage, loss, preds, targets):
        if level == 'step':
            acc, prec, rec = self.acc(preds, targets), self.prec(preds, targets), self.rec(preds, targets)
            self.metric_hist[f'{stage}/acc'].append(acc.item())
            self.metric_hist[f'{stage}/prec'].append(prec.item())
            self.metric_hist[f'{stage}/rec'].append(rec.item())
            self.metric_hist[f'{stage}/loss'].append(loss.item())

            self.log(f"{stage}/loss", loss, prog_bar=True)
            self.log(f"{stage}/acc", acc, prog_bar=True)
            self.log(f"{stage}/prec", prec, prog_bar=False)
            self.log(f"{stage}/rec", rec, prog_bar=False)
        else:
            loss, acc, prec, rec =\
            np.mean(self.metric_hist[f'{stage}/loss']),\
            np.mean(self.metric_hist[f'{stage}/acc']),\
            np.mean(self.metric_hist[f'{stage}/prec']),\
            np.mean(self.metric_hist[f'{stage}/rec'])

            self.log(f"{stage}/loss_epoch", loss)
            self.log(f'{stage}/acc_epoch', acc, self.current_epoch)
            self.log(f'{stage}/prec_epoch', prec, self.current_epoch)
            self.log(f'{stage}/recall_epoch', rec, self.current_epoch)

            for key in self.metric_hist.keys():
                if key.startswith(f'{stage}/'):
                    self.metric_hist[key].clear()

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.logging('step', 'train', loss, preds, targets)
        return {"loss": loss}

    def on_train_epoch_end(self):
        self.logging('epoch', 'train', None, None, None)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.logging('step', 'val', loss, preds, targets)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        self.logging('epoch', 'val', None, None, None)
        
    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.logging('step', 'test', loss, preds, targets)
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch_end(self):
       self.logging('epoch', 'test', None, None, None)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers

        """
        optim_class = getattr(optim, self.hparams.optim_module.target)
        optimizer = optim_class(params=self.model.parameters(), **self.hparams.optim_module.kwargs)

        if self.hparams.lr_sch_module.target is not None:
            lr_sch_class = getattr(lr_scheduler, self.hparams.lr_sch_module.target)
            lr_sch = lr_sch_class(optimizer=optimizer, **self.hparams.lr_sch_module.kwargs)
            assert self.hparams.lr_sch_module.monitor is not None, "Set monitor metric to track by..."
            return {"optimizer": optimizer, "lr_scheduler": lr_sch, "monitor": self.hparams.lr_sch_module.monitor}

        return optimizer
