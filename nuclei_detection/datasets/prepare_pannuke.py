# -*- coding: utf-8 -*-
# Prepare the PanNuke dataset: convert the numpy folds into single png images / npy labels
# and generate the csv/yaml files required by PanNukeDataset.
#
# Original @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
#
# Expected input: the three PanNuke folds, either as downloaded
#   <input_path>/Fold 1/images/fold1/images.npy, <input_path>/Fold 1/images/fold1/types.npy,
#   <input_path>/Fold 1/masks/fold1/masks.npy (and the same for Fold 2, Fold 3)
# or already flattened as <input_path>/fold{0,1,2}/{images,masks,types}.npy
#
# Output (per fold i = 0, 1, 2):
#   <output_path>/fold{i}/images/{i}_{n}.png
#   <output_path>/fold{i}/labels/{i}_{n}.npz      arrays "inst_map" and "type_map" (--label_format npy writes
#                                                 the legacy pickled dict used by CellViT datasets)
#   <output_path>/fold{i}/types.csv               img,type (tissue type)
#   <output_path>/fold{i}/cell_count.csv          Image,Neoplastic,Inflammatory,Connective,Dead,Epithelial
# and in <output_path>: dataset_config.yaml and weight_config.yaml

import argparse
import inspect
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from nuclei_detection.utils.metrics import remap_label  # noqa: E402

NUCLEI_TYPES = ["Background", "Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
TISSUE_TYPES = [
    "Adrenal_gland", "Bile-duct", "Bladder", "Breast", "Cervix", "Colon", "Esophagus",
    "HeadNeck", "Kidney", "Liver", "Lung", "Ovarian", "Pancreatic", "Prostate", "Skin",
    "Stomach", "Testis", "Thyroid", "Uterus",
]


def find_fold_file(input_path: Path, fold: int, name: str) -> Optional[Path]:
    """Locate images.npy / masks.npy / types.npy of a fold in the original or flattened layout"""
    candidates = [
        input_path / f"fold{fold}" / name,
        input_path / f"Fold {fold + 1}" / "images" / f"fold{fold + 1}" / name,
        input_path / f"Fold {fold + 1}" / "masks" / f"fold{fold + 1}" / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # fallback: search recursively
    for pattern in (f"**/fold{fold + 1}/{name}", f"**/Fold {fold + 1}/**/{name}"):
        hits = sorted(input_path.glob(pattern))
        if len(hits) > 0:
            return hits[0]
    return None


def process_fold(fold: int, input_path: Path, output_path: Path, label_format: str = "npz") -> List[str]:
    """Convert one fold. Returns the tissue types of the processed images (for weight_config.yaml)

    Args:
        fold (int): 0-based fold index
        input_path (Path): PanNuke download folder
        output_path (Path): Output dataset folder
        label_format (str, optional): "npz" (arrays, portable across numpy versions) or "npy"
            (legacy pickled dict, as in CellViT datasets). Defaults to "npz".
    """
    output_fold_path = Path(output_path) / f"fold{fold}"
    output_fold_path.mkdir(exist_ok=True, parents=True)
    (output_fold_path / "images").mkdir(exist_ok=True, parents=True)
    (output_fold_path / "labels").mkdir(exist_ok=True, parents=True)

    images_file = find_fold_file(input_path, fold, "images.npy")
    masks_file = find_fold_file(input_path, fold, "masks.npy")
    types_file = find_fold_file(input_path, fold, "types.npy")
    if images_file is None or masks_file is None:
        raise FileNotFoundError(
            f"Could not find images.npy/masks.npy for fold {fold} below {input_path}"
        )

    print(f"Fold: {fold}")
    print("Loading large numpy files, this may take a while")
    images = np.load(images_file)
    masks = np.load(masks_file)
    types = np.load(types_file, allow_pickle=True) if types_file is not None else None
    if types is None:
        print("Warning: types.npy not found, types.csv will not be generated for this fold")

    print("Process images")
    for i in tqdm(range(len(images)), total=len(images)):
        outname = f"{fold}_{i}.png"
        out_img = images[i]
        im = Image.fromarray(out_img.astype(np.uint8))
        im.save(output_fold_path / "images" / outname)

    print("Process masks")
    cell_counts = []
    for i in tqdm(range(len(images)), total=len(images)):
        outname = f"{fold}_{i}.{label_format}"

        # need to create instance map and type map with shape 256x256
        mask = masks[i]
        inst_map = np.zeros((256, 256), dtype=np.int32)
        num_nuc = 0
        counts = []
        for j in range(5):
            # copy value from new array if value is not equal 0
            layer_res = remap_label(mask[:, :, j])
            inst_map = np.where(layer_res != 0, layer_res + num_nuc, inst_map)
            n_layer = int(np.max(layer_res))
            num_nuc = num_nuc + n_layer
            counts.append(n_layer)
        inst_map = remap_label(inst_map)

        type_map = np.zeros((256, 256)).astype(np.int32)
        for j in range(5):
            layer_res = ((j + 1) * np.clip(mask[:, :, j], 0, 1)).astype(np.int32)
            type_map = np.where(layer_res != 0, layer_res, type_map)

        if label_format == "npz":
            np.savez_compressed(output_fold_path / "labels" / outname, inst_map=inst_map, type_map=type_map)
        else:
            outdict = {"inst_map": inst_map, "type_map": type_map}
            np.save(output_fold_path / "labels" / outname, outdict)
        cell_counts.append([f"{fold}_{i}.png"] + counts)

    cell_count_df = pd.DataFrame(cell_counts, columns=["Image"] + NUCLEI_TYPES[1:])
    cell_count_df.to_csv(output_fold_path / "cell_count.csv", index=False)

    fold_types = []
    if types is not None:
        fold_types = [str(t) for t in types]
        types_df = pd.DataFrame(
            {"img": [f"{fold}_{i}.png" for i in range(len(fold_types))], "type": fold_types}
        )
        types_df.to_csv(output_fold_path / "types.csv", index=False)
    return fold_types


def write_dataset_configs(output_path: Path, all_types: List[str]) -> None:
    """Write dataset_config.yaml and weight_config.yaml (tissue counts) into the dataset root"""
    dataset_config = {
        "tissue_types": {t: i for i, t in enumerate(TISSUE_TYPES)},
        "nuclei_types": {t: i for i, t in enumerate(NUCLEI_TYPES)},
    }
    with open(output_path / "dataset_config.yaml", "w") as f:
        yaml.dump(dataset_config, f, sort_keys=False)

    if len(all_types) > 0:
        counts = pd.Series(all_types).value_counts()
        unknown = sorted(set(counts.index) - set(TISSUE_TYPES))
        if unknown:
            print(f"Warning: unknown tissue types in types.npy: {unknown}")
        weight_config = {"tissue": {t: int(counts.get(t, 0)) for t in TISSUE_TYPES}}
        with open(output_path / "weight_config.yaml", "w") as f:
            yaml.dump(weight_config, f, sort_keys=False)


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Convert the PanNuke numpy folds into the folder structure used by NuLite (images, labels, csv and yaml files)",
)
parser.add_argument(
    "--input_path",
    type=str,
    help="Input path of the original PanNuke dataset",
    required=True,
)
parser.add_argument(
    "--output_path",
    type=str,
    help="Output path to store the processed PanNuke dataset",
    required=True,
)
parser.add_argument(
    "--folds",
    type=int,
    nargs="+",
    default=[0, 1, 2],
    help="Folds to process (0-based: 0 -> 'Fold 1', ...)",
)
parser.add_argument(
    "--label_format",
    type=str,
    choices=["npz", "npy"],
    default="npz",
    help="Label file format: npz (plain arrays, portable across numpy versions) or npy (legacy pickled dict)",
)

if __name__ == "__main__":
    opt = parser.parse_args()

    input_path = Path(opt.input_path)
    output_path = Path(opt.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    all_types: List[str] = []
    for fold in opt.folds:
        all_types += process_fold(fold, input_path, output_path, label_format=opt.label_format)
    write_dataset_configs(output_path, all_types)
    print(f"Done. Dataset written to {output_path}")
