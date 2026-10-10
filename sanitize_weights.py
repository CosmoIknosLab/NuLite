# -*- coding: utf-8 -*-
# Convert a training checkpoint into a lightweight inference checkpoint
#
# Checkpoints written by NuLiteTrainer already contain the "config" entry needed by
# CellSegmentationInference. This script is only needed for
#   * old checkpoints saved without "config", or
#   * stripping optimizer/scheduler states to obtain a small file for distribution.
import argparse
from pathlib import Path

import torch
import yaml

from utils.tools import flatten_dict

# Default (PanNuke) configuration used when the checkpoint has no "config" entry
DEFAULT_CONFIG = {
    "data.dataset": "PanNuke",
    "data.num_nuclei_classes": 6,
    "data.num_tissue_classes": 19,
    "training.drop_rate": 0,
    "transformations.normalize.mean": [0.5, 0.5, 0.5],
    "transformations.normalize.std": [0.5, 0.5, 0.5],
    "dataset_config.tissue_types.Adrenal_gland": 0,
    "dataset_config.tissue_types.Bile-duct": 1,
    "dataset_config.tissue_types.Bladder": 2,
    "dataset_config.tissue_types.Breast": 3,
    "dataset_config.tissue_types.Cervix": 4,
    "dataset_config.tissue_types.Colon": 5,
    "dataset_config.tissue_types.Esophagus": 6,
    "dataset_config.tissue_types.HeadNeck": 7,
    "dataset_config.tissue_types.Kidney": 8,
    "dataset_config.tissue_types.Liver": 9,
    "dataset_config.tissue_types.Lung": 10,
    "dataset_config.tissue_types.Ovarian": 11,
    "dataset_config.tissue_types.Pancreatic": 12,
    "dataset_config.tissue_types.Prostate": 13,
    "dataset_config.tissue_types.Skin": 14,
    "dataset_config.tissue_types.Stomach": 15,
    "dataset_config.tissue_types.Testis": 16,
    "dataset_config.tissue_types.Thyroid": 17,
    "dataset_config.tissue_types.Uterus": 18,
    "dataset_config.nuclei_types.Background": 0,
    "dataset_config.nuclei_types.Neoplastic": 1,
    "dataset_config.nuclei_types.Inflammatory": 2,
    "dataset_config.nuclei_types.Connective": 3,
    "dataset_config.nuclei_types.Dead": 4,
    "dataset_config.nuclei_types.Epithelial": 5,
}


def sanitize(checkpoint_path: Path, output_path: Path, arch: str, backbone: str, config_yaml: Path = None) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if config_yaml is not None:
        with open(config_yaml, "r") as f:
            config = flatten_dict(yaml.safe_load(f), sep=".")
    elif checkpoint.get("config") is not None:
        config = dict(checkpoint["config"])
    else:
        config = dict(DEFAULT_CONFIG)

    if backbone is not None:
        config["model.backbone"] = backbone
    if "model.backbone" not in config:
        raise ValueError("The backbone is neither in the checkpoint config nor given via --backbone")

    sanitized = {
        "arch": arch,
        "model_state_dict": checkpoint["model_state_dict"],
        "config": config,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sanitized, output_path)
    print(f"Saved inference checkpoint to {output_path} (backbone: {config['model.backbone']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Strip a training checkpoint down to the entries needed for WSI inference.",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Training checkpoint (e.g. latest_checkpoint.pth)")
    parser.add_argument("--output", type=str, required=True, help="Output path of the inference checkpoint")
    parser.add_argument("--arch", type=str, default="NuLite", help="Architecture name stored in the checkpoint")
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="FastViT backbone (e.g. fastvit_t8, fastvit_s12, fastvit_sa36). Required if the checkpoint has no config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional run configuration yaml (config_0.yaml of the training run) to embed instead of the checkpoint config",
    )
    args = parser.parse_args()
    sanitize(
        checkpoint_path=Path(args.checkpoint),
        output_path=Path(args.output),
        arch=args.arch,
        backbone=args.backbone,
        config_yaml=Path(args.config) if args.config else None,
    )
