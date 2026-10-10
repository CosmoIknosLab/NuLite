# -*- coding: utf-8 -*-
import logging

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ConstantLR, ReduceLROnPlateau

from base_ml.base_trainer import BaseTrainer
from utils.tools import unflatten_dict


class DummyTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trained_epochs = []

    def train_epoch(self, epoch, train_loader, **kwargs):
        self.trained_epochs.append(epoch)
        return {"Loss/Train": 1.0}, {"Example-Predictions/Train": None}

    def validation_epoch(self, epoch, val_dataloader):
        return {"Loss/Validation": 0.5}, {"Example-Predictions/Validation": None}, 0.5

    def train_step(self, batch, batch_idx, num_batches):
        pass

    def validation_step(self, batch, batch_idx):
        pass


def _make_trainer(tmp_path, conf, scheduler_cls=ConstantLR):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    if scheduler_cls is ConstantLR:
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=1)
    else:
        scheduler = scheduler_cls(optimizer)
    return DummyTrainer(
        model=model,
        loss_fn=None,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        logger=logging.getLogger("test"),
        logdir=tmp_path,
        experiment_config=conf,
        mixed_precision=True,
    )


def test_checkpoint_contains_flattened_config_and_resume_is_off_by_none(tmp_path):
    conf = {"model": {"backbone": "fastvit_t8"}, "data": {"num_nuclei_classes": 6}}
    trainer = _make_trainer(tmp_path, conf)
    assert trainer.mixed_precision is False  # float16 autocast is CUDA only

    trainer.save_checkpoint(epoch=4, checkpoint_name="latest_checkpoint.pth")
    checkpoint = torch.load(tmp_path / "checkpoints" / "latest_checkpoint.pth", map_location="cpu", weights_only=False)
    assert checkpoint["arch"] == "Linear"
    assert checkpoint["config"]["model.backbone"] == "fastvit_t8"
    assert unflatten_dict(checkpoint["config"], ".") == conf

    resumed = _make_trainer(tmp_path, conf)
    resumed.resume_checkpoint(checkpoint)
    assert resumed.start_epoch == 5  # epoch 4 was completed, continue with epoch index 5
    resumed.fit(epochs=7, train_dataloader=[], val_dataloader=[])
    assert resumed.trained_epochs == [5, 6]


def test_fit_with_eval_every_and_reduce_lr_on_plateau(tmp_path):
    trainer = _make_trainer(tmp_path, {"a": 1}, scheduler_cls=ReduceLROnPlateau)
    trainer.fit(epochs=3, train_dataloader=[], val_dataloader=[], eval_every=2)
    assert trainer.trained_epochs == [0, 1, 2]
    assert (tmp_path / "checkpoints" / "latest_checkpoint.pth").is_file()
