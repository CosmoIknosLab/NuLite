# -*- coding: utf-8 -*-
# Geometry of the patch grid written by the WSI preprocessing
#
# The preprocessing reads square tiles from one level of the slide pyramid with OpenSlide's
# DeepZoomGenerator (limit_bounds=True). In pixels of that level, tile (row, col) covers
#   rows [row * tile_size - tile_overlap, (row + 1) * tile_size + tile_overlap)
#   cols [col * tile_size - tile_overlap, (col + 1) * tile_size + tile_overlap)
# Border tiles are padded to the full size (top/left padding for the first row/column) and,
# when the target resolution is not a pyramid level, every tile is resized to patch_size.
# PatchGrid inverts this mapping, so that coordinates predicted on a patch can be converted to
# level-0 coordinates of the slide. Kept free of heavy dependencies on purpose: it is shared by
# the preprocessing (which writes the grid to metadata.yaml) and by the inference.

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def patch_to_tile_size(
    patch_size: int, overlap: int, rescaling_factor: float = 1.0
) -> Tuple[int, int]:
    """Given patch size and overlap, it returns the size of the tile of the WSI image.

    The tile size must be reduced on both sides from the overlap, since OpenSlide is adding the overlap to the tile size.
    This function ensures that the resulting patch has the same dimensions as configured and is not extended by the overlap.

    Args:
        patch_size (int): Configured Patch Size (in pixels)
        overlap (int): The amount of overlap between adjecent patches (in pixels, per side)
        rescaling_factor (float, optional): Rescaling of tiles after extraction to generate matching microns per pixel.
            Defaults to 1.0 (no rescaling)

    Returns:
        int: Resulting tile size reduced by overlap (with optional rescaling)
        int: Resulting overlap (with optional rescaling)
    """
    if not math.isclose(rescaling_factor, 1.0, rel_tol=1e-9):
        # sizes read from the slide are rounded up to even numbers
        patch_size = int(math.ceil(rescaling_factor * patch_size / 2) * 2)
        overlap = int(math.ceil(rescaling_factor * overlap / 2) * 2)
    return patch_size - overlap * 2, overlap


@dataclass(frozen=True)
class PatchGrid:
    """Patch grid of a preprocessed WSI.

    Args:
        patch_size (int): Side of the stored patches (pixels, overlap included)
        tile_size (int): Grid step, in pixels of the pyramid level the tiles are read from
        tile_overlap (int): Overlap per side, in pixels of that pyramid level
        downsample (int, optional): Level-0 pixels per pixel of that pyramid level. Defaults to 1.
    """

    patch_size: int
    tile_size: int
    tile_overlap: int
    downsample: int = 1

    @property
    def scale(self) -> float:
        """Level-0 pixels per patch pixel"""
        return self.downsample * (self.tile_size + 2 * self.tile_overlap) / self.patch_size

    def patch_origin(self, row: int, col: int) -> np.ndarray:
        """Level-0 position (row, col order) of the upper-left pixel of patch (row, col)"""
        return self.downsample * np.array(
            [
                row * self.tile_size - self.tile_overlap,
                col * self.tile_size - self.tile_overlap,
            ]
        )

    def to_level0(
        self, points: np.ndarray, row: int, col: int, xy: bool = False
    ) -> np.ndarray:
        """Convert patch pixel coordinates of patch (row, col) to level-0 slide coordinates

        Args:
            points (np.ndarray): Coordinates with shape (..., 2), in (row, col) order or in (x, y) order if xy is True
            row (int): Row of the patch in the grid
            col (int): Column of the patch in the grid
            xy (bool, optional): Points are given in (x, y) order. Defaults to False.

        Returns:
            np.ndarray: Level-0 coordinates, same shape and order as the input (integers if the scale is integral)
        """
        origin = self.patch_origin(row, col)
        if xy:
            origin = origin[::-1]
        points = np.asarray(points)
        scale = self.scale
        if float(scale).is_integer():
            return points * int(scale) + origin
        return points * scale + origin

    @classmethod
    def from_metadata(cls, metadata: dict) -> "PatchGrid":
        """Grid of a preprocessed WSI from its metadata (metadata.yaml)

        Metadata written by older versions of the preprocessing do not contain the grid. It is
        rebuilt exactly as the preprocessing computed it at that time: the overlap per side is half
        of "patch_overlap", the rescaling factor is target_patch_mpp / base_mpp and the pyramid
        level was selected from the bit length of "downsampling".

        Args:
            metadata (dict): WSI metadata

        Returns:
            PatchGrid: Grid used to extract the patches
        """
        patch_size = int(metadata["patch_size"])
        if "tile_size" in metadata and "tile_overlap" in metadata:
            return cls(
                patch_size=patch_size,
                tile_size=int(metadata["tile_size"]),
                tile_overlap=int(metadata["tile_overlap"]),
                downsample=int(metadata["downsampling"]),
            )

        rescaling_factor = 1.0
        base_mpp = metadata.get("base_mpp")
        target_mpp = metadata.get("target_patch_mpp")
        if base_mpp and target_mpp and not math.isclose(base_mpp, target_mpp, rel_tol=1e-9):
            # the stored value is base_mpp * (target_mpp / base_mpp): recover the configured
            # target mpp to get the very same float the preprocessing used
            rescaling_factor = round(float(target_mpp), 6) / float(base_mpp)
        tile_size, tile_overlap = patch_to_tile_size(
            patch_size, int(metadata["patch_overlap"]) // 2, rescaling_factor
        )
        downsample = 2 ** (int(metadata["downsampling"]).bit_length() - 1)
        return cls(
            patch_size=patch_size,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            downsample=downsample,
        )
