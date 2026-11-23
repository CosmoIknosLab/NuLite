# -*- coding: utf-8 -*-
# CellViT Inference Method for Patch-Wise Inference on a patches test set/Whole WSI
#
# Detect Cells with our Networks
# Patches dataset needs to have the follwoing requirements.txt:
# Patch-Size must be 1024, with overlap of 64
#
# We provide preprocessing code here: ./preprocessing/patch_extraction/main_extraction.py
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import os

import argparse
import logging
import uuid
import warnings
from collections import deque
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
import ujson
from einops import rearrange

from shapely import strtree
from shapely.geometry import Polygon, MultiPolygon
from torch.utils.data import DataLoader
from torchvision import transforms as T

from models.nulite import NuLite
from nuclei_detection.datasets.cell_graph_datamodel import CellGraphDataWSI
from nuclei_detection.utils.template_geojson import (
    get_template_point,
    get_template_segmentation,
)
from nuclei_detection.datamodel.wsi_datamodel import WSI
from preprocessing.encoding.datasets.patched_wsi_inference import PatchedWSIInference
from utils.logger import Logger
from utils.patch_grid import PatchGrid
from utils.tools import available_cpu_count, get_size_of_dict, unflatten_dict

try:  # only exists in shapely < 2.1
    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except ImportError:  # pragma: no cover
    pass

# color setup
COLOR_DICT = {
    1: [255, 0, 0],
    2: [34, 221, 77],
    3: [35, 92, 236],
    4: [254, 255, 0],
    5: [255, 159, 68],
}

TYPE_NUCLEI_DICT = {
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}


class CellSegmentationInference:
    def __init__(
            self,
            model_path: Union[Path, str],
            gpu: int,
    ) -> None:
        """Cell Segmentation Inference class.

        After setup, a WSI can be processed by calling process_wsi method

        Args:
            model_path (Union[Path, str]): Path to model checkpoint
            gpu (int): CUDA GPU id to use
            enforce_mixed_precision (bool, optional): Using PyTorch autocasting with dtype float16 to speed up inference. Also good for trained amp networks.
                Can be used to enforce amp inference even for networks trained without amp. Otherwise, the network setting is used.
                Defaults to False.
        """
        self.model_path = Path(model_path)
        if torch.cuda.is_available():
            self.device = f"cuda:{gpu}"
        else:
            self.device = "cpu"
        self.__instantiate_logger()
        if self.device == "cpu":
            self.logger.warning("No CUDA device available - running inference on CPU (slow)")
        self.__load_model()
        self.__load_inference_transforms()

    def __instantiate_logger(self) -> None:
        """Instantiate logger

        Logger is using no formatters. Logs are stored in the run directory under the filename: inference.log
        """
        logger = Logger(
            level="INFO",
        )
        self.logger = logger.create_logger()

    def __load_model(self) -> None:
        """Load model and checkpoint and load the state_dict"""
        self.logger.info(f"Loading model: {self.model_path}")

        model_checkpoint = torch.load(
            self.model_path, map_location="cpu", weights_only=False
        )

        # unpack checkpoint
        if "config" not in model_checkpoint:
            raise KeyError(
                f"Checkpoint {self.model_path} has no 'config' entry. "
                "Checkpoints written by NuLiteTrainer contain it; for older checkpoints use sanitize_weights.py."
            )
        self.run_conf = unflatten_dict(model_checkpoint["config"], ".")
        data_conf = self.run_conf.get("data", {})
        self.model = NuLite(
            num_nuclei_classes=int(data_conf.get("num_nuclei_classes", 6)),
            num_tissue_classes=int(data_conf.get("num_tissue_classes", 19)),
            vit_structure=self.run_conf["model"]["backbone"],
            pretrained_encoder=False,  # weights come from the checkpoint
        )
        self.logger.info(
            self.model.load_state_dict(model_checkpoint["model_state_dict"])
        )
        self.model.eval()
        self.model.reparameterize_encoder()
        self.model.to(self.device)

    def __load_inference_transforms(self):
        """Load the inference transformations from the run_configuration"""
        self.logger.info("Loading inference transformations")

        transform_settings = self.run_conf["transformations"]
        if "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        self.inference_transforms = T.Compose(
            [T.ToTensor(), T.Normalize(mean=mean, std=std)]
        )

    def process_wsi(
            self,
            wsi: WSI,
            subdir_name: str = None,
            batch_size: int = 8,
            geojson: bool = False,
    ) -> None:
        """Process WSI file

        The patch grid is taken from the WSI metadata written by the preprocessing (``metadata.yaml``,
        see :class:`utils.patch_grid.PatchGrid`), so that global coordinates always match the grid the
        patches were extracted with, also with downsampling (pyramid level) and rescaling (target mpp).
        Global coordinates are level-0 pixels of the slide, relative to the origin of the slide bounds
        (``level0_offset`` in the metadata, zero for most formats).

        Args:
            wsi (WSI): WSI object
            subdir_name (str, optional): If provided, a subdir with the given name is created in the cell_detection folder.
                Helpful if you need to store different cell detection results next to each other. Defaults to None (no subdir).
            batch_size (int, optional): Batch-size for inference. Defaults to 8.
            geojson (bool, optional): If a geojson export should be performed. Defaults to False.
        """
        self.logger.info(f"Processing WSI: {wsi.name}")
        wsi_inference_dataset = PatchedWSIInference(
            wsi, transform=self.inference_transforms
        )

        # grid used during preprocessing: maps patch pixels to level-0 slide coordinates
        grid = PatchGrid.from_metadata(wsi.metadata)
        patch_size = grid.patch_size
        # total overlap between neighbouring patches, in patch pixels (margin classification of the cells)
        overlap = int(wsi.metadata["patch_overlap"])
        if grid.scale != 1:
            self.logger.info(
                f"Patch grid: tiles of {grid.tile_size}+2x{grid.tile_overlap} px read with downsampling "
                f"{grid.downsample}, {grid.scale:.4f} level-0 px per patch px"
            )

        num_workers = int(np.clip(int(3 / 4 * available_cpu_count()), 1, 2 * batch_size))

        wsi_inference_dataloader = DataLoader(
            dataset=wsi_inference_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=wsi_inference_dataset.collate_batch,
            pin_memory=False,
        )
        dataset_config = self.run_conf["dataset_config"]
        nuclei_types = dataset_config["nuclei_types"]

        if subdir_name is not None:
            outdir = Path(wsi.patched_slide_path) / "cell_detection" / subdir_name
        else:
            outdir = Path(wsi.patched_slide_path) / "cell_detection"
        outdir.mkdir(exist_ok=True, parents=True)

        cell_dict_wsi = []  # for storing all cell information
        cell_dict_detection = []  # for storing only the centroids

        graph_data = {
            "cell_tokens": [],
            "positions": [],
            "contours": [],
            "metadata": {"wsi_metadata": wsi.metadata, "nuclei_types": nuclei_types},
        }
        processed_patches = []

        memory_usage = 0  # bytes
        cell_count = 0
        use_amp = self.device.startswith("cuda")

        with torch.no_grad():

            pbar = tqdm.tqdm(total=len(wsi_inference_dataset), desc="Patches")

            for batch in wsi_inference_dataloader:
                patches = batch[0].to(self.device)

                metadata = batch[1]

                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=use_amp
                ):
                    predictions = self.model.forward(patches, retrieve_tokens=True)

                # reshape, apply softmax to segmentation maps
                instance_types, tokens = self.get_cell_predictions_with_tokens(
                    predictions, magnification=wsi.metadata["magnification"]
                )
                # size (in input pixels) of one token of the bottleneck feature map
                token_stride = patch_size / tokens.shape[-1]
                num_tokens_h, num_tokens_w = tokens.shape[-2], tokens.shape[-1]

                # unpack each patch from batch
                for idx, (patch_instance_types, patch_metadata) in enumerate(
                        zip(instance_types, metadata)
                ):
                    pbar.update(1)
                    processed_patches.append(
                        f"{patch_metadata['row']}_{patch_metadata['col']}"
                    )

                    # level-0 position (row, col) of the upper-left pixel of the patch
                    patch_row, patch_col = patch_metadata["row"], patch_metadata["col"]
                    offset_global = grid.patch_origin(patch_row, patch_col)

                    # extract cell information
                    for cell in patch_instance_types.values():
                        if cell["type"] == nuclei_types["Background"]:
                            continue
                        # centroid and contour are (x, y), bbox is (row, col)
                        centroid_global = grid.to_level0(
                            cell["centroid"], patch_row, patch_col, xy=True
                        )
                        contour_global = grid.to_level0(
                            cell["contour"], patch_row, patch_col, xy=True
                        )
                        bbox_global = grid.to_level0(cell["bbox"], patch_row, patch_col)
                        cell_dict = {
                            "bbox": bbox_global.tolist(),
                            "centroid": centroid_global.tolist(),
                            "contour": contour_global.tolist(),
                            "type_prob": cell["type_prob"],
                            "type": cell["type"],
                            "patch_coordinates": [
                                patch_metadata["row"],
                                patch_metadata["col"],
                            ],
                            "cell_status": get_cell_position_marging(
                                cell["bbox"], patch_size, overlap
                            ),
                            "offset_global": offset_global.tolist()
                        }
                        cell_detection = {
                            "bbox": bbox_global.tolist(),
                            "centroid": centroid_global.tolist(),
                            "type": cell["type"],
                        }
                        if np.max(cell["bbox"]) == patch_size or np.min(cell["bbox"]) == 0:
                            position = get_cell_position(cell["bbox"], patch_size)
                            cell_dict["edge_position"] = True
                            cell_dict["edge_information"] = {}
                            cell_dict["edge_information"]["position"] = position
                            cell_dict["edge_information"][
                                "edge_patches"
                            ] = get_edge_patch(
                                position, patch_metadata["row"], patch_metadata["col"]
                            )
                        else:
                            cell_dict["edge_position"] = False

                        cell_dict_wsi.append(cell_dict)
                        cell_dict_detection.append(cell_detection)

                        # get the cell token: average the bottleneck tokens covered by the cell bbox.
                        # bbox is [[rmin, cmin], [rmax, cmax]] in patch pixels -> convert to token indices
                        r0, c0 = np.floor(cell["bbox"][0] / token_stride).astype(int)
                        r1, c1 = np.ceil(cell["bbox"][1] / token_stride).astype(int)
                        r0 = int(np.clip(r0, 0, num_tokens_h - 1))
                        c0 = int(np.clip(c0, 0, num_tokens_w - 1))
                        r1 = int(np.clip(r1, r0 + 1, num_tokens_h))  # at least one token
                        c1 = int(np.clip(c1, c0 + 1, num_tokens_w))
                        cell_token = tokens[idx, :, r0:r1, c0:c1]
                        cell_token = torch.mean(
                            rearrange(cell_token, "D H W -> (H W) D"), dim=0
                        )

                        graph_data["cell_tokens"].append(cell_token)
                        graph_data["positions"].append(torch.Tensor(centroid_global))
                        graph_data["contours"].append(torch.Tensor(contour_global))

                        cell_count = cell_count + 1
                        memory_usage += get_size_of_dict(cell_dict) + get_size_of_dict(cell_detection)
                        memory_usage += (
                            cell_token.nelement() * cell_token.element_size()
                            + centroid_global.nbytes
                            + contour_global.nbytes
                        )

                pbar.set_postfix(Cells=cell_count, Memory=f"{memory_usage / 1024 ** 2:.2f} MB")

            pbar.close()

        # post processing
        self.logger.info(f"Detected cells before cleaning: {len(cell_dict_wsi)}")
        if len(cell_dict_wsi) > 0:
            keep_idx = self.post_process_edge_cells(cell_list=cell_dict_wsi)
        else:
            self.logger.warning(f"No cells detected in WSI {wsi.name}")
            keep_idx = []
        cell_dict_wsi = [cell_dict_wsi[idx_c] for idx_c in keep_idx]
        cell_dict_detection = [cell_dict_detection[idx_c] for idx_c in keep_idx]
        graph_data["cell_tokens"] = [
            graph_data["cell_tokens"][idx_c] for idx_c in keep_idx
        ]
        graph_data["positions"] = [graph_data["positions"][idx_c] for idx_c in keep_idx]
        graph_data["contours"] = [graph_data["contours"][idx_c] for idx_c in keep_idx]
        self.logger.info(f"Detected cells after cleaning: {len(keep_idx)}")

        self.logger.info(
            f"Processed all patches. Storing final results: {str(outdir / 'cells.json')} and cell_detection.json"
        )
        cell_dict_wsi = {
            "wsi_metadata": wsi.metadata,
            "processed_patches": processed_patches,
            "type_map": dataset_config["nuclei_types"],
            "cells": cell_dict_wsi,
        }
        with open(str(outdir / "cells.json"), "w") as outfile:
            ujson.dump(cell_dict_wsi, outfile, indent=2)
        if geojson:
            self.logger.info("Converting segmentation to geojson")
            geojson_list = self.convert_geojson(cell_dict_wsi["cells"], True)
            with open(str(outdir / "cells.geojson"), "w") as outfile:
                ujson.dump(geojson_list, outfile, indent=2)

        cell_dict_detection = {
            "wsi_metadata": wsi.metadata,
            "processed_patches": processed_patches,
            "type_map": dataset_config["nuclei_types"],
            "cells": cell_dict_detection,
        }
        with open(str(outdir / "cell_detection.json"), "w") as outfile:
            ujson.dump(cell_dict_detection, outfile, indent=2)
        if geojson:
            self.logger.info("Converting detection to geojson")
            geojson_list = self.convert_geojson(cell_dict_detection["cells"], False)
            with open(str(outdir / "cell_detection.geojson"), "w") as outfile:
                ujson.dump(geojson_list, outfile, indent=2)

        self.logger.info(
            f"Create cell graph with embeddings and save it under: {str(outdir / 'cells.pt')}"
        )
        if len(graph_data["cell_tokens"]) > 0:
            x = torch.stack(graph_data["cell_tokens"])
            positions = torch.stack(graph_data["positions"])
        else:
            x = torch.empty((0, tokens.shape[1]))
            positions = torch.empty((0, 2))
        graph = CellGraphDataWSI(
            x=x,
            positions=positions,
            contours=graph_data["contours"],
            metadata=graph_data["metadata"],
        )
        torch.save(graph, outdir / "cells.pt")

        nuclei_types_inverse = {v: k for k, v in nuclei_types.items()}
        if len(cell_dict_wsi["cells"]) > 0:
            cell_stats = pd.DataFrame(cell_dict_wsi["cells"])["type"].value_counts().to_dict()
        else:
            cell_stats = {}
        verbose_stats = {nuclei_types_inverse[k]: int(v) for k, v in cell_stats.items()}
        self.logger.info(f"Finished with cell detection for WSI {wsi.name}")
        self.logger.info("Stats:")
        self.logger.info(f"{verbose_stats}")

    def get_cell_predictions_with_tokens(
            self, predictions: dict, magnification: int = 40
    ) -> Tuple[List[dict], torch.Tensor]:
        """Take the raw predictions, apply softmax and calculate type instances

        Args:
            predictions (dict): Network predictions with tokens. Keys:
            magnification (int, optional): WSI magnification. Defaults to 40.

        Returns:
            Tuple[List[dict], torch.Tensor]:
                * List[dict]: List with a dictionary for each batch element with cell seg results
                    Contains bbox, contour, 2D-position, type and type_prob for each cell
                * List[dict]: Network tokens on cpu device with shape (batch_size, num_tokens_h, num_tokens_w, embd_dim)
        """
        predictions["nuclei_binary_map"] = F.softmax(
            predictions["nuclei_binary_map"], dim=1
        )  # shape: (batch_size, 2, H, W)
        predictions["nuclei_type_map"] = F.softmax(
            predictions["nuclei_type_map"], dim=1
        )  # shape: (batch_size, num_nuclei_classes, H, W)
        # get the instance types
        (
            _,
            instance_types,
        ) = self.model.calculate_instance_map(predictions, magnification=magnification)

        tokens = predictions["tokens"].to("cpu")

        return instance_types, tokens

    def post_process_edge_cells(self, cell_list: List[dict]) -> List[int]:
        """Use the CellPostProcessor to remove multiple cells and merge due to overlap

        Args:
            cell_list (List[dict]): List with cell-dictionaries. Required keys:
                * bbox
                * centroid
                * contour
                * type_prob
                * type
                * patch_coordinates
                * cell_status
                * offset_global

        Returns:
            List[int]: List with integers of cells that should be kept
        """
        cell_processor = CellPostProcessor(cell_list, self.logger)
        cleaned_cells = cell_processor.post_process_cells()

        return list(cleaned_cells.index.values)

    def convert_geojson(
            self, cell_list: List[dict], polygons: bool = False
    ) -> List[dict]:
        """Convert a list of cells to a geojson object

        Either a segmentation object (polygon) or detection points are converted

        Args:
            cell_list (list[dict]): Cell list with dict entry for each cell.
                Required keys for detection:
                    * type
                    * centroid
                Required keys for segmentation:
                    * type
                    * contour
            polygons (bool, optional): If polygon segmentations (True) or detection points (False). Defaults to False.

        Returns:
            List[dict]: Geojson like list
        """
        if len(cell_list) == 0:
            return []
        if polygons:
            cell_segmentation_df = pd.DataFrame(cell_list)
            detected_types = sorted(cell_segmentation_df.type.unique())
            geojson_placeholder = []
            for cell_type in detected_types:
                cells = cell_segmentation_df[cell_segmentation_df["type"] == cell_type]
                contours = cells["contour"].to_list()
                final_c = []
                for c in contours:
                    # close the ring on a copy: do not mutate the contours stored in the cell dicts
                    final_c.append([list(c) + [c[0]]])

                cell_geojson_object = get_template_segmentation()
                cell_geojson_object["id"] = str(uuid.uuid4())
                cell_geojson_object["geometry"]["coordinates"] = final_c
                cell_geojson_object["properties"]["classification"][
                    "name"
                ] = TYPE_NUCLEI_DICT[cell_type]
                cell_geojson_object["properties"]["classification"][
                    "color"
                ] = COLOR_DICT[cell_type]
                geojson_placeholder.append(cell_geojson_object)
        else:
            cell_detection_df = pd.DataFrame(cell_list)
            detected_types = sorted(cell_detection_df.type.unique())
            geojson_placeholder = []
            for cell_type in detected_types:
                cells = cell_detection_df[cell_detection_df["type"] == cell_type]
                centroids = cells["centroid"].to_list()
                cell_geojson_object = get_template_point()
                cell_geojson_object["id"] = str(uuid.uuid4())
                cell_geojson_object["geometry"]["coordinates"] = centroids
                cell_geojson_object["properties"]["classification"][
                    "name"
                ] = TYPE_NUCLEI_DICT[cell_type]
                cell_geojson_object["properties"]["classification"][
                    "color"
                ] = COLOR_DICT[cell_type]
                geojson_placeholder.append(cell_geojson_object)
        return geojson_placeholder


class CellPostProcessor:
    def __init__(self, cell_list: List[dict], logger: logging.Logger) -> None:
        """POst-Processing a list of cells from one WSI

        Args:
            cell_list (List[dict]): List with cell-dictionaries. Required keys:
                * bbox
                * centroid
                * contour
                * type_prob
                * type
                * patch_coordinates
                * cell_status
                * offset_global
            logger (logging.Logger): Logger
        """
        self.logger = logger if logger is not None else logging.getLogger("__main__")
        self.logger.info("Initializing Cell-Postprocessor")
        self.cell_df = pd.DataFrame(cell_list)
        # string representation "row_col" of the patch position for fast querying
        coords = self.cell_df["patch_coordinates"]
        self.cell_df["patch_row"] = coords.map(lambda c: c[0])
        self.cell_df["patch_col"] = coords.map(lambda c: c[1])
        self.cell_df["patch_coordinates"] = coords.map(lambda c: f"{c[0]}_{c[1]}")

        self.mid_cells = self.cell_df[
            self.cell_df["cell_status"] == 0
            ]  # cells in the mid
        self.cell_df_margin = self.cell_df[
            self.cell_df["cell_status"] != 0
            ]  # cells either touching the border or margin

    def post_process_cells(self) -> pd.DataFrame:
        """Main Post-Processing coordinator, entry point

        Returns:
            pd.DataFrame: DataFrame with post-processed and cleaned cells
        """
        #self.logger.info("Finding edge-cells for merging")
        cleaned_edge_cells = self._clean_edge_cells()
        #self.logger.info("Removal of cells detected multiple times")
        cleaned_edge_cells = self._remove_overlap(cleaned_edge_cells)

        # merge with mid cells
        postprocessed_cells = pd.concat(
            [self.mid_cells, cleaned_edge_cells]
        ).sort_index()
        return postprocessed_cells

    def _clean_edge_cells(self) -> pd.DataFrame:
        """Create a DataFrame that just contains all margin cells (cells inside the margin, not touching the border)
        and border/edge cells (touching border) with no overlapping equivalent (e.g, if patch has no neighbour)

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """

        margin_cells = self.cell_df_margin[
            self.cell_df_margin["edge_position"] == 0
            ]  # cells at the margin, but not touching the border
        edge_cells = self.cell_df_margin[
            self.cell_df_margin["edge_position"] == 1
            ]  # cells touching the border
        existing_patches = set(self.cell_df["patch_coordinates"].to_list())

        def _neighbour_missing(edge_information: dict) -> bool:
            # the first listed neighbour is the patch sharing the touched border
            edge_patch = edge_information["edge_patches"][0]
            return f"{edge_patch[0]}_{edge_patch[1]}" not in existing_patches

        # cells touching the border without having an overlap from other patches
        if len(edge_cells) > 0:
            keep = edge_cells["edge_information"].map(_neighbour_missing).astype(bool)
            edge_cells_unique = edge_cells[keep.values]
        else:
            edge_cells_unique = edge_cells

        cleaned_edge_cells = pd.concat([margin_cells, edge_cells_unique])

        return cleaned_edge_cells.sort_index()

    def _remove_overlap(self, cleaned_edge_cells: pd.DataFrame) -> pd.DataFrame:
        """Remove overlapping cells from provided DataFrame

        Args:
            cleaned_edge_cells (pd.DataFrame): DataFrame that should be cleaned

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        merged_cells = cleaned_edge_cells
        max_iterations = 20
        overlaps = 0

        for iteration in range(max_iterations):
            poly_list = []  # shapely polygons
            uid_list = []  # dataframe index of each polygon (same order as poly_list)
            for idx, cell_info in merged_cells.iterrows():
                poly = Polygon(cell_info["contour"])
                if not poly.is_valid:
                    poly = _fix_polygon(poly)
                poly_list.append(poly)
                uid_list.append(idx)

            # use an strtree for fast querying
            tree = strtree.STRtree(poly_list)

            merged_idx = deque()
            iterated_cells = set()
            overlaps = 0

            for query_pos, query_poly in enumerate(poly_list):
                query_uid = uid_list[query_pos]
                if query_uid in iterated_cells:
                    continue
                # candidate indices (this also contains the self-intersection)
                candidate_pos = _strtree_query(tree, query_poly)
                submergers = []  # (uid, polygon) of all cells that overlap with query
                for pos in candidate_pos:
                    inter_uid = uid_list[pos]
                    if inter_uid == query_uid or inter_uid in iterated_cells:
                        continue
                    inter_poly = poly_list[pos]
                    inter_area = query_poly.intersection(inter_poly).area
                    if (
                            inter_area / max(query_poly.area, 1e-8) > 0.01
                            or inter_area / max(inter_poly.area, 1e-8) > 0.01
                    ):
                        overlaps = overlaps + 1
                        submergers.append((inter_uid, inter_poly))
                        iterated_cells.add(inter_uid)
                if len(submergers) == 0:
                    # no (strong enough) overlap, just add
                    merged_idx.append(query_uid)
                else:
                    # merging strategy: take the biggest cell (query included)
                    submergers.append((query_uid, query_poly))
                    selected = max(submergers, key=lambda t: t[1].area)
                    merged_idx.append(selected[0])
                iterated_cells.add(query_uid)

            self.logger.info(
                f"Iteration {iteration}: Found overlap of # cells: {overlaps}"
            )
            merged_cells = cleaned_edge_cells.loc[
                cleaned_edge_cells.index.isin(merged_idx)
            ].sort_index()
            if overlaps == 0:
                self.logger.info("Found all overlapping cells")
                break
        else:
            self.logger.warning(
                f"Not all doubled cells removed after {max_iterations} iterations, still {overlaps} to remove. "
                "For performance reasons, we stop iterations now."
            )

        return merged_cells.sort_index()


def _fix_polygon(poly: Polygon) -> Polygon:
    """Fix an invalid polygon with buffer(0); for MultiPolygons keep the largest part.

    Works with shapely 1.8 and 2.x (uses ``.geoms``).
    """
    fixed = poly.buffer(0)
    if isinstance(fixed, MultiPolygon):
        parts = list(fixed.geoms)
        if len(parts) == 0:
            return poly
        return max(parts, key=lambda p: p.area)
    if isinstance(fixed, Polygon):
        return fixed
    # buffer(0) may return a GeometryCollection or an empty geometry: keep original
    return poly


def _strtree_query(tree: "strtree.STRtree", geom) -> List[int]:
    """Query an STRtree and always return the positional indices of the hits.

    shapely < 2 returns geometries from ``query`` (indices via ``query_items``),
    shapely >= 2 returns indices directly.
    """
    if hasattr(tree, "query_items"):  # shapely 1.8
        return list(tree.query_items(geom))
    return [int(i) for i in np.asarray(tree.query(geom)).ravel()]


def get_cell_position(bbox: np.ndarray, patch_size: int) -> List[int]:
    """Get cell position as a list

    Entry is 1, if cell touches the border: [top, right, down, left]

    Args:
        bbox (np.ndarray): Bounding-Box of cell
        patch_size (int, optional): Patch-size. Defaults to 1024.

    Returns:
        List[int]: List with 4 integers for each position
    """
    # bbox = 2x2 array in h, w style
    # bbox[0,0] = upper position (height)
    # bbox[1,0] = lower dimension (height)
    # boox[0,1] = left position (width)
    # bbox[1,1] = right position (width)
    # bbox[:,0] -> x dimensions
    top, left, down, right = False, False, False, False
    if bbox[0, 0] == 0:
        top = True
    if bbox[0, 1] == 0:
        left = True
    if bbox[1, 0] == patch_size:
        down = True
    if bbox[1, 1] == patch_size:
        right = True
    position = [top, right, down, left]
    position = [int(pos) for pos in position]

    return position


def get_cell_position_marging(
        bbox: np.ndarray, patch_size: int = 1024, margin: int = 64
) -> int:
    """Get the status of the cell, describing the cell position

    A cell is either in the mid (0) or at one of the borders (1-8)

    # Numbers are assigned clockwise, starting from top left
    # i.e., top left = 1, top = 2, top right = 3, right = 4, bottom right = 5 bottom = 6, bottom left = 7, left = 8
    # Mid status is denoted by 0

    Args:
        bbox (np.ndarray): Bounding Box of cell
        patch_size (int, optional): Patch-Size. Defaults to 1024.
        margin (int, optional): Margin-Size. Defaults to 64.

    Returns:
        int: Cell Status
    """
    cell_status = None
    if np.max(bbox) > patch_size - margin or np.min(bbox) < margin:
        if bbox[0, 0] < margin:
            # top left, top or top right
            if bbox[0, 1] < margin:
                # top left
                cell_status = 1
            elif bbox[1, 1] > patch_size - margin:
                # top right
                cell_status = 3
            else:
                # top
                cell_status = 2
        elif bbox[1, 1] > patch_size - margin:
            # top right, right or bottom right
            if bbox[1, 0] > patch_size - margin:
                # bottom right
                cell_status = 5
            else:
                # right
                cell_status = 4
        elif bbox[1, 0] > patch_size - margin:
            # bottom right, bottom, bottom left
            if bbox[0, 1] < margin:
                # bottom left
                cell_status = 7
            else:
                # bottom
                cell_status = 6
        elif bbox[0, 1] < margin:
            # bottom left, left, top left, but only left is left
            cell_status = 8
    else:
        cell_status = 0

    return cell_status


def get_edge_patch(position: List[int], row: int, col: int) -> List[List[int]]:
    """Get the neighbouring patches a border cell may continue into.

    Args:
        position (List[int]): Border flags as returned by get_cell_position: [top, right, down, left]
        row (int): Patch row
        col (int): Patch col

    Returns:
        List[List[int]]: List of [row, col] neighbour patches. The directly adjacent patches
            (sharing a touched border) come first, diagonal patches (for corners) last.
            Always non-empty when at least one border is touched.
    """
    top, right, down, left = (bool(p) for p in position)
    neighbours = []
    if top:
        neighbours.append([row - 1, col])
    if right:
        neighbours.append([row, col + 1])
    if down:
        neighbours.append([row + 1, col])
    if left:
        neighbours.append([row, col - 1])
    # diagonal neighbours for corner cells
    if top and right:
        neighbours.append([row - 1, col + 1])
    if down and right:
        neighbours.append([row + 1, col + 1])
    if down and left:
        neighbours.append([row + 1, col - 1])
    if top and left:
        neighbours.append([row - 1, col - 1])
    return neighbours


# CLI
class InferenceWSIParser:
    """Parser"""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform NuLite inference on a preprocessed (patched) WSI or on a whole dataset of patched WSIs.",
        )
        requiredNamed = parser.add_argument_group("required named arguments")
        requiredNamed.add_argument(
            "--model",
            type=str,
            help="Model checkpoint file that is used for inference",
            required=True,
        )
        parser.add_argument(
            "--gpu", type=int, help="Cuda-GPU ID for inference", default=0
        )
        parser.add_argument(
            "--magnification",
            type=float,
            help="Network magnification. Is used for checking patch magnification such that we use the correct resolution for network.",
            default=40,
        )

        parser.add_argument(
            "--batch_size",
            type=int,
            help="Inference batch-size",
            default=8,
        )
        parser.add_argument(
            "--outdir_subdir",
            type=str,
            help="If provided, a subdir with the given name is created in the cell_detection folder where the results are stored. Default: None",
            default=None,
        )
        parser.add_argument(
            "--geojson",
            action="store_true",
            help="Set this flag to export results as additional geojson files for loading them into Software like QuPath.",
        )

        # subparsers for either loading a WSI or a WSI folder

        # WSI
        subparsers = parser.add_subparsers(
            dest="command",
            description="Main run command for either performing inference on single WSI-file or on whole dataset",
        )
        subparser_wsi = subparsers.add_parser(
            "process_wsi", description="Process a single WSI file"
        )
        subparser_wsi.add_argument(
            "--wsi_path",
            type=str,
            help="Path to WSI file",
            required=True,
        )
        subparser_wsi.add_argument(
            "--patched_slide_path",
            type=str,
            help="Path to patched WSI file (specific WSI file, not parent path of patched slide dataset)",
            required=True,
        )

        # Dataset
        subparser_dataset = subparsers.add_parser(
            "process_dataset",
            description="Process a whole dataset",
        )
        subparser_dataset.add_argument(
            "--wsi_paths",
            type=str,
            help="Path to the folder where all WSI are stored",
            required=True,
        )
        subparser_dataset.add_argument(
            "--patch_dataset_path",
            type=str,
            help="Path to the folder where the patch dataset is stored",
            required=True,
        )
        subparser_dataset.add_argument(
            "--filelist",
            type=str,
            help="Filelist with WSI to process. Must be a .csv file with one row denoting the filenames (named 'Filename')."
                 "If not provided, all WSI files with given ending in the filelist are processed.",
            default=None,
        )
        subparser_dataset.add_argument(
            "--wsi_extension",
            type=str,
            help="The extension types used for the WSI files, see configs.python.config (WSI_EXT)",
            default="svs",
        )

        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)


def check_wsi(wsi: WSI, magnification: float = 40.0):
    """Check if provided patched WSI is having the right settings

    Args:
        wsi (WSI): WSI to check
        magnification (float, optional): Check magnification. Defaults to 40.0.

    Raises:
        RuntimeError: The magnification is not matching to the network input magnification.
        RuntimeError: The patch-size is not divisible by 256.
        RuntimeError: The patch-size is not 1024 or 2048
        RuntimeError: The overlap is not 64 or 128 px
    """
    if wsi.metadata.get("magnification") is not None:
        patch_magnification = float(wsi.metadata["magnification"])
    else:
        patch_magnification = float(
            float(wsi.metadata["base_magnification"]) / wsi.metadata["downsampling"]
        )
    patch_size = int(wsi.metadata["patch_size"])
    patch_overlap = int(wsi.metadata["patch_overlap"])

    if not np.isclose(patch_magnification, magnification):
        raise RuntimeError(
            f"The patch magnification ({patch_magnification}) is not matching to the network input magnification ({magnification})."
        )
    if (patch_size % 256) != 0:
        raise RuntimeError("The patch-size must be divisible by 256.")
    if patch_size not in [1024, 2048]:
        raise RuntimeError(f"The patch-size must be 1024 or 2048, got {patch_size}.")
    if patch_overlap not in [64, 128]:
        raise RuntimeError(f"The patch-overlap must be 64 or 128, got {patch_overlap}.")
