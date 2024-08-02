# -*- coding: utf-8 -*-
# Coordinate the datasets, used to select the right dataset with corresponding setting
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from typing import Callable

from torch.utils.data import Dataset

from nuclei_detection.datasets.pannuke import PanNukeDataset


def select_dataset(
    dataset_name: str, split: str, dataset_config: dict, transforms: Callable = None
) -> Dataset:
    """Select a cell segmentation dataset from the provided ones, currently just PanNuke is implemented here

    Args:
        dataset_name (str): Name of dataset to use.
            Must be one of: [pannuke, lizzard]
        split (str): Split to use.
            Must be one of: ["train", "val", "validation", "test"]
        dataset_config (dict): Dictionary with dataset configuration settings
        transforms (Callable, optional): PyTorch Image and Mask transformations. Defaults to None.

    Raises:
        NotImplementedError: Unknown dataset

    Returns:
        Dataset: Cell segmentation dataset
    """
    assert split.lower() in [
        "train",
        "val",
        "validation",
        "test",
    ], "Unknown split type!"

    if dataset_name.lower() == "pannuke":
        if split == "train":
            folds = dataset_config["train_folds"]
        if split == "val" or split == "validation":
            folds = dataset_config["val_folds"]
        if split == "test":
            folds = dataset_config["test_folds"]
        dataset = PanNukeDataset(
            dataset_path=dataset_config["dataset_path"],
            folds=folds,
            transforms=transforms,
            stardist=dataset_config.get("stardist", False),
            regression=dataset_config.get("regression_loss", False),
            cache_dataset=True,
        )
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset_name}")
    return dataset
