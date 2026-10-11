# -*- coding: utf-8 -*-
# Base Trainer Class
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import logging
from abc import abstractmethod
from typing import Tuple, Union

import torch
import torch.nn as nn
from base_ml.base_early_stopping import EarlyStopping
from pathlib import Path
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from utils.tools import flatten_dict


def _create_grad_scaler(enabled: bool):
    """Create a GradScaler, compatible with torch < 2.3 (torch.cuda.amp) and >= 2.3 (torch.amp)."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - old torch versions
        return torch.cuda.amp.GradScaler(enabled=enabled)


class BaseTrainer:
    """
    Base class for all trainers with important ML components

    Args:
        model (nn.Module):  Model that should be trained
        loss_fn (_Loss): Loss function
        optimizer (Optimizer): Optimizer
        scheduler (_LRScheduler): Learning rate scheduler
        device (str): Cuda device to use, e.g., cuda:0.
        logger (logging.Logger): Logger module
        logdir (Union[Path, str]): Logging directory
        experiment_config (dict): Configuration of this experiment
        early_stopping (EarlyStopping, optional): Early Stopping Class. Defaults to None.
        accum_iter (int, optional): Accumulation steps for gradient accumulation.
            Provide a number greater than 1 for activating gradient accumulation. Defaults to 1.
        mixed_precision (bool, optional): If mixed-precision should be used (autocast + GradScaler).
            Only effective on CUDA devices. Defaults to False.
        log_images (bool, optional): If example images should be generated. Defaults to False.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: _Loss,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        device: str,
        logger: logging.Logger,
        logdir: Union[Path, str],
        experiment_config: dict,
        early_stopping: EarlyStopping = None,
        accum_iter: int = 1,
        mixed_precision: bool = False,
        log_images: bool = False,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.logdir = Path(logdir)
        self.early_stopping = early_stopping
        self.accum_iter = accum_iter
        self.start_epoch = 0
        self.experiment_config = experiment_config
        self.log_images = log_images
        # float16 autocast is only supported on CUDA
        self.mixed_precision = bool(mixed_precision) and str(device).startswith("cuda")
        self.scaler = _create_grad_scaler(enabled=self.mixed_precision)

    @abstractmethod
    def train_epoch(
        self, epoch: int, train_loader: DataLoader, **kwargs
    ) -> Tuple[dict, dict]:
        """Training logic for a training epoch

        Args:
            epoch (int): Current epoch number
            train_loader (DataLoader): Train dataloader

        Raises:
            NotImplementedError: Needs to be implemented

        Returns:
            Tuple[dict, dict]: wandb logging dictionaries
                * Scalar metrics
                * Image metrics
        """
        raise NotImplementedError

    @abstractmethod
    def validation_epoch(
        self, epoch: int, val_dataloader: DataLoader
    ) -> Tuple[dict, dict, float]:
        """Training logic for an validation epoch

        Args:
            epoch (int): Current epoch number
            val_dataloader (DataLoader): Validation dataloader

        Raises:
            NotImplementedError: Needs to be implemented

        Returns:
            Tuple[dict, dict, float]: wandb logging dictionaries and early_stopping_metric
                * Scalar metrics
                * Image metrics
                * Early Stopping metric as float
        """
        raise NotImplementedError

    @abstractmethod
    def train_step(self, batch: object, batch_idx: int, num_batches: int):
        """Training logic for one training batch

        Args:
            batch (object): A training batch
            batch_idx (int): Current batch index
            num_batches (int): Maximum number of batches

        Raises:
            NotImplementedError: Needs to be implemented
        """

        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch, batch_idx: int):
        """Training logic for one validation batch

        Args:
            batch (object): A training batch
            batch_idx (int): Current batch index

        Raises:
            NotImplementedError: Needs to be implemented
        """

    def fit(
        self,
        epochs: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        metric_init: dict = None,
        eval_every: int = 1,
        **kwargs,
    ):
        """Fitting function to start training and validation of the trainer

        Args:
            epochs (int): Number of epochs the network should be training
            train_dataloader (DataLoader): Dataloader with training data
            val_dataloader (DataLoader): Dataloader with validation data
            metric_init (dict, optional): Initialization dictionary with scalar metrics that should be initialized for startup.
                This is just import for logging with wandb if you want to have the plots properly scaled.
                The data in the the metric dictionary is used as values for epoch 0 (before training has startetd).
                If not provided, step 0 (epoch 0) is not logged. Should have the same scalar keys as training and validation epochs report.
                For more information, you should have a look into the train_epoch and val_epoch methods where the wandb logging dicts are assembled.
                Defaults to None.
            eval_every (int, optional): How often the network should be evaluated (after how many epochs). Defaults to 1.
            **kwargs
        """

        self.logger.info(f"Starting training, total number of epochs: {epochs}")
        for epoch in range(self.start_epoch, epochs):
            # training epoch
            self.logger.info(f"Epoch: {epoch+1}/{epochs}")
            train_scalar_metrics, train_image_metrics = self.train_epoch(
                epoch, train_dataloader, **kwargs
            )
            self.logger.debug(f"Train metrics: {train_scalar_metrics}")
            self.save_image_metrics(train_image_metrics, epoch)

            val_scalar_metrics = None
            early_stopping_metric = None
            if ((epoch + 1) % eval_every) == 0:
                # validation epoch
                (
                    val_scalar_metrics,
                    val_image_metrics,
                    early_stopping_metric,
                ) = self.validation_epoch(epoch, val_dataloader)
                self.logger.debug(f"Validation metrics: {val_scalar_metrics}")
                self.save_image_metrics(val_image_metrics, epoch)

            # log learning rate
            curr_lr = self.optimizer.param_groups[0]["lr"]
            if early_stopping_metric is not None and self.early_stopping is not None:
                # early stopping
                best_model = self.early_stopping(early_stopping_metric, epoch)
                if best_model:
                    self.logger.info("New best model - save checkpoint")
                    self.save_checkpoint(epoch, "model_best.pth")
                elif self.early_stopping.early_stop:
                    self.logger.info("Performing early stopping!")
                    self.save_checkpoint(epoch, "latest_checkpoint.pth")
                    break
            self.save_checkpoint(epoch, "latest_checkpoint.pth")

            # scheduling
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if val_scalar_metrics is not None:
                    self.scheduler.step(float(val_scalar_metrics["Loss/Validation"]))
            else:
                self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.debug(f"Old lr: {curr_lr:.6f} - New lr: {new_lr:.6f}")

    def save_image_metrics(self, image_metrics: dict, epoch: int) -> None:
        """Store example figures (if any) under logdir/example_images (replaces the former wandb image logging)

        Args:
            image_metrics (dict): Dictionary name -> matplotlib figure (or None)
            epoch (int): Current epoch (0-based)
        """
        if not image_metrics:
            return
        figures = {k: v for k, v in image_metrics.items() if v is not None}
        if len(figures) == 0:
            return
        import matplotlib.pyplot as plt

        image_dir = self.logdir / "example_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            safe_name = name.replace("/", "_").replace(" ", "_")
            fig.savefig(image_dir / f"epoch{epoch + 1:03d}_{safe_name}.png")
            plt.close(fig)

    def save_checkpoint(self, epoch: int, checkpoint_name: str):
        if self.early_stopping is None:
            best_metric = None
            best_epoch = None
        else:
            # plain Python types: numpy scalars pickled under numpy 2 cannot be loaded with numpy 1.x
            best_metric = (
                float(self.early_stopping.best_metric)
                if self.early_stopping.best_metric is not None
                else None
            )
            best_epoch = (
                int(self.early_stopping.best_epoch)
                if self.early_stopping.best_epoch is not None
                else None
            )

        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            # flattened run configuration ("a.b.c" keys): needed by CellSegmentationInference to rebuild the model
            "config": flatten_dict(self.experiment_config, sep=".")
            if self.experiment_config is not None
            else None,
            "logdir": str(self.logdir.resolve()),
            "run_name": str(Path(self.logdir).name),
            "scaler_state_dict": self.scaler.state_dict()
            if self.scaler is not None
            else None,
        }

        checkpoint_dir = self.logdir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        filename = str(checkpoint_dir / checkpoint_name)
        torch.save(state, filename)

    def resume_checkpoint(self, checkpoint):
        self.logger.info("Loading checkpoint")
        self.logger.info("Loading Model")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.logger.info("Loading Optimizer state dict")
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.early_stopping is not None:
            self.early_stopping.best_metric = checkpoint["best_metric"]
            self.early_stopping.best_epoch = checkpoint["best_epoch"]
        if self.scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # checkpoint["epoch"] is the (0-based) index of the last completed epoch
        completed_epoch = int(checkpoint["epoch"])
        self.logger.info(f"Checkpoint epoch: {completed_epoch + 1}")
        self.start_epoch = completed_epoch + 1
        self.logger.info(f"Next epoch is: {self.start_epoch + 1}")
