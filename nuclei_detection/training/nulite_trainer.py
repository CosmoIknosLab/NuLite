# -*- coding: utf-8 -*-
# NuLite Trainer Class (adapted from the CellViT trainer)
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from matplotlib import pyplot as plt
from skimage.color import rgba2rgb
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from base_ml.base_early_stopping import EarlyStopping
from base_ml.base_trainer import BaseTrainer

from models.nulite import NuLite
from nuclei_detection.utils.tools import cropping_center
from utils.tools import AverageMeter, DataclassHVStorage


class NuLiteTrainer(BaseTrainer):
    """NuLite trainer class

    Args:
        model (NuLite): NuLite model that should be trained
        loss_fn_dict (dict): Dictionary with loss functions for each branch with a dictionary of loss functions.
            Name of branch as top-level key, followed by a dictionary with loss name, loss fn and weighting factor
            Example:
            {
                "nuclei_binary_map": {"bce": {loss_fn(Callable), weight_factor(float)}, "dice": {loss_fn(Callable), weight_factor(float)}},
                "hv_map": {"bce": {loss_fn(Callable), weight_factor(float)}, "dice": {loss_fn(Callable), weight_factor(float)}},
                "nuclei_type_map": {"bce": {loss_fn(Callable), weight_factor(float)}, "dice": {loss_fn(Callable), weight_factor(float)}}
                "tissue_types": {"ce": {loss_fn(Callable), weight_factor(float)}}
            }
            Required Keys are:
                * nuclei_binary_map
                * hv_map
                * nuclei_type_map
                * tissue types
        optimizer (Optimizer): Optimizer
        scheduler (_LRScheduler): Learning rate scheduler
        device (str): Cuda device to use, e.g., cuda:0.
        logger (logging.Logger): Logger module
        logdir (Union[Path, str]): Logging directory
        num_classes (int): Number of nuclei classes
        dataset_config (dict): Dataset configuration. Required Keys are:
            * "tissue_types": describing the present tissue types with corresponding integer
            * "nuclei_types": describing the present nuclei types with corresponding integer
        experiment_config (dict): Configuration of this experiment
        early_stopping (EarlyStopping, optional):  Early Stopping Class. Defaults to None.
        log_images (bool, optional): If example images should be generated. Defaults to False.
        magnification (int, optional): Image magnification. Please select either 40 or 20. Defaults to 40.
        mixed_precision (bool, optional): If mixed-precision (autocast + GradScaler) should be used. Defaults to True.
    """

    def __init__(
            self,
            model: NuLite,
            loss_fn_dict: dict,
            optimizer: Optimizer,
            scheduler: LRScheduler,
            device: str,
            logger: logging.Logger,
            logdir: Union[Path, str],
            num_classes: int,
            dataset_config: dict,
            experiment_config: dict,
            early_stopping: EarlyStopping = None,
            log_images: bool = False,
            magnification: int = 40,
            mixed_precision: bool = True,
    ):
        super().__init__(
            model=model,
            loss_fn=None,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            logdir=logdir,
            experiment_config=experiment_config,
            early_stopping=early_stopping,
            accum_iter=1,
            mixed_precision=mixed_precision,
            log_images=log_images,
        )
        self.loss_fn_dict = loss_fn_dict
        self.num_classes = num_classes
        self.dataset_config = dataset_config
        self.tissue_types = dataset_config["tissue_types"]
        self.reverse_tissue_types = {v: k for k, v in self.tissue_types.items()}
        self.nuclei_types = dataset_config["nuclei_types"]
        self.magnification = magnification

        # setup logging objects
        self.loss_avg_tracker = {"Total_Loss": AverageMeter("Total_Loss", ":.4f")}
        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                self.loss_avg_tracker[f"{branch}_{loss_name}"] = AverageMeter(
                    f"{branch}_{loss_name}", ":.4f"
                )
        self.batch_avg_tissue_acc = AverageMeter("Batch_avg_tissue_ACC", ":.4f")

    def _reset_trackers(self) -> None:
        """Reset all running averages (called at the beginning of each epoch)"""
        self.loss_avg_tracker["Total_Loss"].reset()
        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                self.loss_avg_tracker[f"{branch}_{loss_name}"].reset()
        self.batch_avg_tissue_acc.reset()

    def train_epoch(
            self, epoch: int, train_dataloader: DataLoader, unfreeze_epoch: int = 50
    ) -> Tuple[dict, dict]:
        """Training logic for a training epoch

        Args:
            epoch (int): Current epoch number
            train_dataloader (DataLoader): Train dataloader
            unfreeze_epoch (int, optional): Epoch to unfreeze layers (kept for config compatibility, unused)
        Returns:
            Tuple[dict, dict]: logging dictionaries
                * Scalar metrics
                * Image metrics
        """
        self.model.train()

        train_example_img = None
        self._reset_trackers()

        # randomly select a batch that should be displayed
        if self.log_images:
            select_example_image = int(torch.randint(0, len(train_dataloader), (1,)))
        else:
            select_example_image = None
        train_loop = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))

        for batch_idx, batch in train_loop:
            return_example_images = batch_idx == select_example_image
            example_img = self.train_step(
                batch,
                batch_idx,
                len(train_dataloader),
                return_example_images=return_example_images,
            )
            if example_img is not None:
                train_example_img = example_img
            train_loop.set_postfix(
                {
                    "Loss": np.round(self.loss_avg_tracker["Total_Loss"].avg, 3),
                }
            )

        scalar_metrics = {
            "Loss/Train": self.loss_avg_tracker["Total_Loss"].avg,
        }

        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                scalar_metrics[f"{branch}_{loss_name}/Train"] = self.loss_avg_tracker[
                    f"{branch}_{loss_name}"
                ].avg

        self.logger.info(
            f"{'Training epoch stats:' : <25} "
            f"Loss: {self.loss_avg_tracker['Total_Loss'].avg:.4f} - "
        )

        image_metrics = {"Example-Predictions/Train": train_example_img}

        return scalar_metrics, image_metrics

    def train_step(self, batch: object, batch_idx: int, num_batches: int, return_example_images: bool) -> Union[
        plt.Figure, None]:
        """Training step

        Args:
            batch (object): Training batch, consisting of images ([0]), masks ([1]), tissue_types ([2]) and figure filenames ([3])
            batch_idx (int): Batch index
            num_batches (int): Total number of batches in epoch
            return_example_images (bool): If an example prediction image should be returned

        Returns:
            Union[plt.Figure, None]: Example prediction image (if requested)
        """
        # unpack batch
        imgs = batch[0].to(self.device)  # imgs shape: (batch_size, 3, H, W)
        masks = batch[1]  # dict: keys: "instance_map", "nuclei_map", "nuclei_binary_map", "hv_map"
        tissue_types = batch[2]  # list[str]

        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=self.mixed_precision
        ):
            # make predictions
            predictions_ = self.model.forward(imgs)
            # reshaping and postprocessing
            predictions = self.unpack_predictions(predictions=predictions_)
            gt = self.unpack_masks(masks=masks, tissue_types=tissue_types)
            # calculate loss
            total_loss = self.calculate_loss(predictions, gt)

        # backward pass (outside the autocast region, as recommended by PyTorch)
        self.scaler.scale(total_loss).backward()

        if ((batch_idx + 1) % self.accum_iter == 0) or ((batch_idx + 1) == num_batches):
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        if return_example_images:
            return self._generate_example_image_safe(imgs, predictions, gt)
        return None

    def validation_epoch(
            self, epoch: int, val_dataloader: DataLoader
    ) -> Tuple[dict, dict, float]:
        """Validation logic for a validation epoch

        Args:
            epoch (int): Current epoch number
            val_dataloader (DataLoader): Validation dataloader

        Returns:
            Tuple[dict, dict, float]: logging dictionaries
                * Scalar metrics
                * Image metrics
                * Early stopping metric (validation loss)
        """
        self.model.eval()
        val_example_img = None
        self._reset_trackers()

        # randomly select a batch that should be displayed
        if self.log_images:
            select_example_image = int(torch.randint(0, len(val_dataloader), (1,)))
        else:
            select_example_image = None

        val_loop = tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader))

        with torch.no_grad():
            for batch_idx, batch in val_loop:
                return_example_images = batch_idx == select_example_image
                example_img = self.validation_step(
                    batch, batch_idx, return_example_images
                )
                if example_img is not None:
                    val_example_img = example_img
                val_loop.set_postfix(
                    {
                        "Loss": np.round(self.loss_avg_tracker["Total_Loss"].avg, 3),
                    }
                )

        scalar_metrics = {
            "Loss/Validation": self.loss_avg_tracker["Total_Loss"].avg,
        }

        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                scalar_metrics[
                    f"{branch}_{loss_name}/Validation"
                ] = self.loss_avg_tracker[f"{branch}_{loss_name}"].avg

        self.logger.info(
            f"{'Validation epoch stats:' : <25} "
            f"Loss: {self.loss_avg_tracker['Total_Loss'].avg:.4f} - "
        )

        image_metrics = {"Example-Predictions/Validation": val_example_img}
        return scalar_metrics, image_metrics, self.loss_avg_tracker["Total_Loss"].avg

    def validation_step(self, batch: object, batch_idx: int, return_example_images: bool):
        """Validation step

        Args:
            batch (object): Validation batch, consisting of images ([0]), masks ([1]), tissue_types ([2]) and figure filenames ([3])
            batch_idx (int): Batch index
            return_example_images (bool): If an example prediction image should be returned

        Returns:
            Union[plt.Figure, None]: Example prediction image (if requested)
        """
        # unpack batch, for shape compare train_step method
        imgs = batch[0].to(self.device)
        masks = batch[1]
        tissue_types = batch[2]

        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=self.mixed_precision
        ):
            # make predictions
            predictions_ = self.model.forward(imgs)
            # reshaping and postprocessing
            predictions = self.unpack_predictions(predictions=predictions_)
            gt = self.unpack_masks(masks=masks, tissue_types=tissue_types)
            # calculate loss
            _ = self.calculate_loss(predictions, gt)

        if return_example_images:
            return self._generate_example_image_safe(imgs, predictions, gt)
        return None

    def _generate_example_image_safe(
            self, imgs: torch.Tensor, predictions: DataclassHVStorage, gt: DataclassHVStorage
    ) -> Union[plt.Figure, None]:
        """Compute instance maps and generate the example figure; never let plotting break the training."""
        try:
            self._add_instance_maps(predictions)
            return self.generate_example_image(
                imgs, predictions, gt, num_images=4, num_nuclei_classes=self.num_classes
            )
        except Exception as exc:  # plotting is best-effort
            self.logger.error(f"Could not generate example image, continuing without it: {exc}")
            return None

    def _add_instance_maps(self, predictions: DataclassHVStorage) -> None:
        """Run the (CPU) post-processing to add instance maps to the predictions (needed for plotting only)

        Args:
            predictions (DataclassHVStorage): Predictions with softmax outputs
        """
        pred_dict = {
            "nuclei_binary_map": predictions.nuclei_binary_map.detach().float(),
            "nuclei_type_map": predictions.nuclei_type_map.detach().float(),
            "hv_map": predictions.hv_map.detach().float(),
        }
        instance_map, instance_types = self.model.calculate_instance_map(
            pred_dict, magnification=self.magnification
        )
        predictions.instance_map = instance_map
        predictions.instance_types = instance_types
        predictions.instance_types_nuclei = self.model.generate_instance_nuclei_map(
            instance_map, instance_types
        )

    def unpack_predictions(self, predictions: dict) -> DataclassHVStorage:
        """Unpack the given predictions. Main focus lays on reshaping and postprocessing predictions, e.g. separating instances

        Args:
            predictions (dict): Dictionary with the following keys:
                * tissue_types: Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
                * nuclei_binary_map: Logit output for binary nuclei prediction branch. Shape: (batch_size, 2, H, W)
                * hv_map: Logit output for hv-prediction. Shape: (batch_size, 2, H, W)
                * nuclei_type_map: Logit output for nuclei instance-prediction. Shape: (batch_size, num_nuclei_classes, H, W)

        Returns:
            DataclassHVStorage: Processed network output
        """
        predictions["nuclei_binary_map"] = F.softmax(
            predictions["nuclei_binary_map"], dim=1
        )  # shape: (batch_size, 2, H, W)
        predictions["nuclei_type_map"] = F.softmax(
            predictions["nuclei_type_map"], dim=1
        )  # shape: (batch_size, num_nuclei_classes, H, W)

        predictions = DataclassHVStorage(
            nuclei_binary_map=predictions["nuclei_binary_map"],
            hv_map=predictions["hv_map"],
            nuclei_type_map=predictions["nuclei_type_map"],
            tissue_types=predictions["tissue_types"],
            instance_map=None,
            instance_types=None,
            instance_types_nuclei=None,
            batch_size=predictions["tissue_types"].shape[0],
            regression_map=predictions.get("regression_map", None),
            num_nuclei_classes=self.num_classes,
        )

        return predictions

    def unpack_masks(self, masks: dict, tissue_types: list) -> DataclassHVStorage:
        """Unpack the given masks. Main focus lays on reshaping and postprocessing masks to generate one dict

        Args:
            masks (dict): Required keys are:
                * instance_map: Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W)
                * nuclei_binary_map: Binary nuclei segmentations. Shape: (batch_size, H, W)
                * hv_map: HV-Map. Shape: (batch_size, 2, H, W)
                * nuclei_type_map: Nuclei type map (integer class per pixel). Shape: (batch_size, H, W)

            tissue_types (list): List of string names of ground-truth tissue types

        Returns:
            DataclassHVStorage: GT-Results with matching shapes and output types
        """
        # get ground truth values, perform one hot encoding for segmentation maps
        gt_nuclei_binary_map_onehot = (
            F.one_hot(masks["nuclei_binary_map"], num_classes=2)
        ).type(
            torch.float32
        )  # background, nuclei
        # (B, H, W) -> do not squeeze: a batch of size 1 would lose its batch dimension
        nuclei_type_maps = masks["nuclei_type_map"].type(torch.int64)
        gt_nuclei_type_maps_onehot = F.one_hot(
            nuclei_type_maps, num_classes=self.num_classes
        ).type(
            torch.float32
        )  # background + nuclei types

        # assemble ground truth dictionary
        gt = {
            "nuclei_type_map": gt_nuclei_type_maps_onehot.permute(0, 3, 1, 2).to(
                self.device
            ),  # shape: (batch_size, num_nuclei_classes, H, W)
            "nuclei_binary_map": gt_nuclei_binary_map_onehot.permute(0, 3, 1, 2).to(
                self.device
            ),  # shape: (batch_size, 2, H, W)
            "hv_map": masks["hv_map"].to(self.device),  # shape: (batch_size, 2, H, W)
            "instance_map": masks["instance_map"].to(
                self.device
            ),  # shape: (batch_size, H, W) -> each instance has one integer
            "instance_types_nuclei": (
                    gt_nuclei_type_maps_onehot * masks["instance_map"][..., None]
            )
            .permute(0, 3, 1, 2)
            .to(
                self.device
            ),  # shape: (batch_size, num_nuclei_classes, H, W) -> instance has one integer, for each nuclei class
            "tissue_types": torch.Tensor([self.tissue_types[t] for t in tissue_types])
            .type(torch.LongTensor)
            .to(self.device),  # shape: batch_size
        }
        if "regression_map" in masks:
            gt["regression_map"] = masks["regression_map"].to(self.device)

        gt = DataclassHVStorage(
            **gt,
            batch_size=gt["tissue_types"].shape[0],
            num_nuclei_classes=self.num_classes,
        )
        return gt

    def calculate_loss(
            self, predictions: DataclassHVStorage, gt: DataclassHVStorage
    ) -> torch.Tensor:
        """Calculate the loss

        Args:
            predictions (DataclassHVStorage): Predictions
            gt (DataclassHVStorage): Ground-Truth values

        Returns:
            torch.Tensor: Loss
        """
        predictions = predictions.get_dict()
        gt = gt.get_dict()

        total_loss = 0

        for branch, pred in predictions.items():
            if branch in [
                "instance_map",
                "instance_types",
                "instance_types_nuclei",
            ]:
                continue
            if branch not in self.loss_fn_dict:
                continue
            branch_loss_fns = self.loss_fn_dict[branch]
            for loss_name, loss_setting in branch_loss_fns.items():
                loss_fn = loss_setting["loss_fn"]
                weight = loss_setting["weight"]
                if loss_name == "msge":
                    loss_value = loss_fn(
                        input=pred,
                        target=gt[branch],
                        focus=gt["nuclei_binary_map"],
                        device=self.device,
                    )
                else:
                    loss_value = loss_fn(input=pred, target=gt[branch])
                if torch.isnan(loss_value).any():
                    self.logger.warning(f"NaN loss encountered: {branch}/{loss_name}")
                total_loss = total_loss + weight * loss_value
                self.loss_avg_tracker[f"{branch}_{loss_name}"].update(
                    loss_value.detach().cpu().numpy()
                )
        self.loss_avg_tracker["Total_Loss"].update(total_loss.detach().cpu().numpy())

        return total_loss

    @staticmethod
    def generate_example_image(
            imgs: Union[torch.Tensor, np.ndarray],
            predictions: DataclassHVStorage,
            gt: DataclassHVStorage,
            num_nuclei_classes: int,
            num_images: int = 2,
    ) -> plt.Figure:
        """Generate example plot with image, binary_pred, hv-map and instance map from prediction and ground-truth

        Args:
            imgs (Union[torch.Tensor, np.ndarray]): Images to process, a random number (num_images) is selected from this stack
                Shape: (batch_size, 3, H', W')
            predictions (DataclassHVStorage): Predictions. instance_map must be set (see _add_instance_maps)
            gt (DataclassHVStorage): gt
            num_nuclei_classes (int): Number of total nuclei classes including background
            num_images (int, optional): Number of example patches to display. Defaults to 2.

        Returns:
            plt.Figure: Figure with example patches
        """
        predictions = predictions.get_dict()
        gt = gt.get_dict()

        assert predictions["instance_map"] is not None, "Instance maps must be computed before plotting"
        num_images = int(min(num_images, imgs.shape[0]))

        predictions["nuclei_binary_map"] = predictions["nuclei_binary_map"].permute(
            0, 2, 3, 1
        )
        predictions["hv_map"] = predictions["hv_map"].permute(0, 2, 3, 1)
        predictions["nuclei_type_map"] = predictions["nuclei_type_map"].permute(
            0, 2, 3, 1
        )

        gt["nuclei_binary_map"] = gt["nuclei_binary_map"].permute(0, 2, 3, 1)
        gt["hv_map"] = gt["hv_map"].permute(0, 2, 3, 1)
        gt["nuclei_type_map"] = gt["nuclei_type_map"].permute(0, 2, 3, 1)

        h = gt["hv_map"].shape[1]
        w = gt["hv_map"].shape[2]

        sample_indices = torch.randint(0, imgs.shape[0], (num_images,))
        # convert to rgb and crop to selection
        sample_images = (
            imgs[sample_indices].permute(0, 2, 3, 1).contiguous().cpu().float().numpy()
        )  # convert to rgb
        sample_images = cropping_center(sample_images, (h, w), True)
        # images are normalized: rescale each one to [0, 1] for display
        img_min = sample_images.min(axis=(1, 2, 3), keepdims=True)
        img_max = sample_images.max(axis=(1, 2, 3), keepdims=True)
        sample_images = (sample_images - img_min) / (img_max - img_min + 1e-10)

        # get predictions
        pred_sample_binary_map = (
            predictions["nuclei_binary_map"][sample_indices, :, :, 1]
            .detach()
            .cpu()
            .float()
            .numpy()
        )
        pred_sample_hv_map = (
            predictions["hv_map"][sample_indices].detach().cpu().float().numpy()
        )
        pred_sample_instance_maps = (
            predictions["instance_map"][sample_indices].detach().cpu().numpy()
        )
        pred_sample_type_maps = (
            torch.argmax(predictions["nuclei_type_map"][sample_indices], dim=-1)
            .detach()
            .cpu()
            .numpy()
        )

        # get ground truth labels
        gt_sample_binary_map = (
            gt["nuclei_binary_map"][sample_indices, :, :, 1].detach().cpu().numpy()
        )
        gt_sample_hv_map = gt["hv_map"][sample_indices].detach().cpu().numpy()
        gt_sample_instance_map = (
            gt["instance_map"][sample_indices].detach().cpu().numpy()
        )
        gt_sample_type_map = (
            torch.argmax(gt["nuclei_type_map"][sample_indices], dim=-1)
            .detach()
            .cpu()
            .numpy()
        )

        # create colormaps
        hv_cmap = plt.get_cmap("jet")
        binary_cmap = plt.get_cmap("jet")
        instance_map = plt.get_cmap("viridis")

        def _normalize(arr: np.ndarray) -> np.ndarray:
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-10)

        # setup plot
        fig, axs = plt.subplots(num_images, figsize=(6, 2 * num_images), dpi=150)
        axs = np.atleast_1d(axs)

        for i in range(num_images):
            placeholder = np.zeros((2 * h, 6 * w, 3))
            # orig image
            placeholder[:h, :w, :3] = sample_images[i]
            placeholder[h: 2 * h, :w, :3] = sample_images[i]
            # binary prediction
            placeholder[:h, w: 2 * w, :3] = rgba2rgb(
                binary_cmap(gt_sample_binary_map[i])
            )
            placeholder[h: 2 * h, w: 2 * w, :3] = rgba2rgb(
                binary_cmap(pred_sample_binary_map[i])
            )
            # hv maps
            placeholder[:h, 2 * w: 3 * w, :3] = rgba2rgb(
                hv_cmap((gt_sample_hv_map[i, :, :, 0] + 1) / 2)
            )
            placeholder[h: 2 * h, 2 * w: 3 * w, :3] = rgba2rgb(
                hv_cmap((pred_sample_hv_map[i, :, :, 0] + 1) / 2)
            )
            placeholder[:h, 3 * w: 4 * w, :3] = rgba2rgb(
                hv_cmap((gt_sample_hv_map[i, :, :, 1] + 1) / 2)
            )
            placeholder[h: 2 * h, 3 * w: 4 * w, :3] = rgba2rgb(
                hv_cmap((pred_sample_hv_map[i, :, :, 1] + 1) / 2)
            )
            # instance_predictions
            placeholder[:h, 4 * w: 5 * w, :3] = rgba2rgb(
                instance_map(_normalize(gt_sample_instance_map[i]))
            )
            placeholder[h: 2 * h, 4 * w: 5 * w, :3] = rgba2rgb(
                instance_map(_normalize(pred_sample_instance_maps[i]))
            )
            # type_predictions
            placeholder[:h, 5 * w: 6 * w, :3] = rgba2rgb(
                binary_cmap(gt_sample_type_map[i] / num_nuclei_classes)
            )
            placeholder[h: 2 * h, 5 * w: 6 * w, :3] = rgba2rgb(
                binary_cmap(pred_sample_type_maps[i] / num_nuclei_classes)
            )

            # plotting
            axs[i].imshow(placeholder)
            axs[i].set_xticks([], [])

            # plot labels in first row
            if i == 0:
                axs[i].set_xticks(np.arange(w / 2, 6 * w, w))
                axs[i].set_xticklabels(
                    [
                        "Image",
                        "Binary-Cells",
                        "HV-Map-0",
                        "HV-Map-1",
                        "Cell Instances",
                        "Nuclei-Instances",
                    ],
                    fontsize=6,
                )
                axs[i].xaxis.tick_top()

            axs[i].set_yticks(np.arange(h / 2, 2 * h, h))
            axs[i].set_yticklabels(["GT", "Pred."], fontsize=6)
            axs[i].tick_params(axis="both", which="both", length=0)
            grid_x = np.arange(w, 6 * w, w)
            grid_y = np.arange(h, 2 * h, h)

            for x_seg in grid_x:
                axs[i].axvline(x_seg, color="black")
            for y_seg in grid_y:
                axs[i].axhline(y_seg, color="black")

        fig.suptitle(f"Patch Predictions for {num_images} Examples")

        fig.tight_layout()

        return fig
