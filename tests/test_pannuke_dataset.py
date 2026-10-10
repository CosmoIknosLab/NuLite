# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from nuclei_detection.datasets import prepare_pannuke
from nuclei_detection.datasets.dataset_coordinator import select_dataset
from nuclei_detection.datasets.pannuke import PanNukeDataset

TISSUES = ["Breast", "Colon", "Lung"]


def _synthetic_fold(rng, n_images=6):
    """Images/masks/types in the original PanNuke format (masks: 256x256x6, instance ids per class channel)"""
    images = np.zeros((n_images, 256, 256, 3), dtype=np.uint8)
    masks = np.zeros((n_images, 256, 256, 6), dtype=np.float64)
    types = []
    yy, xx = np.mgrid[0:256, 0:256]
    for i in range(n_images):
        images[i] = 220
        inst_id = 0
        for _ in range(rng.integers(3, 10)):
            cls = int(rng.integers(0, 5))
            cy, cx, r = rng.integers(15, 240), rng.integers(15, 240), rng.integers(5, 10)
            disk = ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r) & (masks[i, :, :, :5].sum(-1) == 0)
            if disk.sum() < 10:
                continue
            inst_id += 1
            masks[i, :, :, cls][disk] = inst_id
            images[i][disk] = (90, 50, 130)
        masks[i, :, :, 5] = (masks[i, :, :, :5].sum(-1) == 0).astype(np.float64)
        types.append(TISSUES[i % len(TISSUES)])
    return images, masks, np.array(types, dtype=object)


@pytest.fixture(scope="module")
def prepared_dataset(tmp_path_factory):
    rng = np.random.default_rng(0)
    raw = tmp_path_factory.mktemp("pannuke_raw")
    out = tmp_path_factory.mktemp("pannuke_prepared")
    for fold in range(2):
        images, masks, types = _synthetic_fold(rng)
        img_dir = raw / f"Fold {fold + 1}" / "images" / f"fold{fold + 1}"
        mask_dir = raw / f"Fold {fold + 1}" / "masks" / f"fold{fold + 1}"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        np.save(img_dir / "images.npy", images)
        np.save(img_dir / "types.npy", types)
        np.save(mask_dir / "masks.npy", masks)
    all_types = []
    for fold in range(2):
        all_types += prepare_pannuke.process_fold(fold, raw, out)
    prepare_pannuke.write_dataset_configs(out, all_types)
    return out


def test_prepare_writes_all_required_files(prepared_dataset):
    for fold in range(2):
        fold_dir = prepared_dataset / f"fold{fold}"
        assert len(list((fold_dir / "images").glob("*.png"))) == 6
        assert len(list((fold_dir / "labels").glob("*.npy"))) == 6
        types = pd.read_csv(fold_dir / "types.csv")
        assert list(types.columns) == ["img", "type"] and len(types) == 6
        counts = pd.read_csv(fold_dir / "cell_count.csv")
        assert list(counts.columns) == ["Image", "Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
        label = np.load(fold_dir / "labels" / f"{fold}_0.npy", allow_pickle=True)[()]
        inst_ids = np.unique(label["inst_map"])
        assert inst_ids[0] == 0 and np.array_equal(inst_ids[1:], np.arange(1, len(inst_ids)))  # contiguous ids
        assert int(counts.iloc[0, 1:].sum()) == len(inst_ids) - 1
    with open(prepared_dataset / "dataset_config.yaml") as f:
        dataset_config = yaml.safe_load(f)
    assert dataset_config["nuclei_types"]["Epithelial"] == 5
    assert dataset_config["tissue_types"]["Uterus"] == 18
    with open(prepared_dataset / "weight_config.yaml") as f:
        weight_config = yaml.safe_load(f)
    assert sum(weight_config["tissue"].values()) == 12


def test_dataset_items_and_sampling_weights(prepared_dataset):
    dataset = select_dataset(
        "pannuke",
        split="Train",  # case-insensitive
        dataset_config={"train_folds": [0], "dataset_path": str(prepared_dataset)},
    )
    assert isinstance(dataset, PanNukeDataset)
    assert len(dataset) == 6
    img, masks, tissue, name = dataset[0]
    assert img.shape == (3, 256, 256) and img.dtype == torch.float32
    assert float(img.max()) <= 1.0  # scaled to [0, 1] without transforms
    assert masks["instance_map"].shape == (256, 256)
    assert masks["nuclei_type_map"].max() <= 5
    assert set(masks["nuclei_binary_map"].unique().tolist()) <= {0, 1}
    assert masks["hv_map"].shape == (2, 256, 256)
    assert -1.0 <= float(masks["hv_map"].min()) and float(masks["hv_map"].max()) <= 1.0
    assert tissue in TISSUES and name == "0_0.png"

    dataset.load_cell_count()
    weights = dataset.get_sampling_weights_cell_tissue(gamma=0.85)
    assert weights.shape == (6,)
    assert torch.isfinite(weights).all() and (weights > 0).all()
