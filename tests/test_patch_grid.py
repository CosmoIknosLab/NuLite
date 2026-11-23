# -*- coding: utf-8 -*-
import numpy as np
import pytest
from PIL import Image

from utils.patch_grid import PatchGrid, patch_to_tile_size

# metadata.yaml of the example slide shipped with the repository (written before the grid was stored)
LEGACY_METADATA = {
    "base_magnification": 40.0,
    "downsampling": 1,
    "patch_overlap": 64,
    "patch_size": 1024,
    "base_mpp": 0.2485,
    "target_patch_mpp": 0.2485,
    "magnification": 40.0,
}


def _former_origin(row, col, patch_size=1024, overlap=64, downsampling=1):
    """Patch offset computed by the former inference code (correct only without downsampling/rescaling)"""
    return np.array(
        [
            int(row * patch_size * downsampling - (row + 0.5) * overlap),
            int(col * patch_size * downsampling - (col + 0.5) * overlap),
        ]
    )


def test_grid_reproduces_former_coordinates_without_rescaling():
    grid = PatchGrid.from_metadata(LEGACY_METADATA)
    assert grid == PatchGrid(patch_size=1024, tile_size=960, tile_overlap=32, downsample=1)
    assert grid.scale == 1
    # metadata written by the current preprocessing describe the same grid
    assert PatchGrid.from_metadata({**LEGACY_METADATA, "tile_size": 960, "tile_overlap": 32}) == grid
    contour = np.array([[10, 20], [30, 25], [12, 40]])  # (x, y) as returned by the post-processing
    bbox = np.array([[20, 10], [40, 30]])  # (row, col)
    for row in range(0, 40, 3):
        for col in range(0, 40, 7):
            offset = _former_origin(row, col)
            assert np.array_equal(grid.patch_origin(row, col), offset)
            mapped = grid.to_level0(contour, row, col, xy=True)
            assert np.array_equal(mapped, contour + np.flip(offset))
            assert np.issubdtype(mapped.dtype, np.integer)
            assert np.array_equal(grid.to_level0(bbox, row, col), bbox + offset)


def test_grid_from_legacy_metadata_with_downsampling_and_rescaling():
    # target_mpp = 0.5 on a 0.25 mpp slide: pyramid level with downsampling 2, no rescaling
    meta = {**LEGACY_METADATA, "downsampling": 2, "base_mpp": 0.25, "target_patch_mpp": 0.25}
    grid = PatchGrid.from_metadata(meta)
    assert grid == PatchGrid(1024, 960, 32, 2)
    assert grid.scale == 2
    assert np.array_equal(grid.patch_origin(3, 5), 2 * np.array([3 * 960 - 32, 5 * 960 - 32]))
    assert np.array_equal(grid.to_level0(np.array([10, 20]), 0, 0), np.array([-64 + 20, -64 + 40]))

    # former rescaling (target_mpp 0.27 on a 0.2485 slide): rebuilt with the same float factor
    rescaling = 0.27 / 0.2485
    meta = {**LEGACY_METADATA, "target_patch_mpp": 0.2485 * rescaling}
    grid = PatchGrid.from_metadata(meta)
    tile_size, tile_overlap = patch_to_tile_size(1024, 32, rescaling)
    assert (tile_size, tile_overlap) == (1042, 36)
    assert grid == PatchGrid(1024, tile_size, tile_overlap, 1)
    assert grid.scale == pytest.approx(1114 / 1024)
    assert grid.to_level0(np.array([512, 512]), 0, 0).dtype.kind == "f"

    # the former code selected the level from the bit length of a non power of two downsampling
    assert PatchGrid.from_metadata({**LEGACY_METADATA, "downsampling": 3}).downsample == 2


def _dot_slide(height=2400, width=2896, pitch=60, jitter=15, radius=4, seed=0):
    """White slide with dark square dots at known, irregular positions (level-0 pixel indices)"""
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    centres = []
    for y in range(pitch // 2, height - pitch // 2, pitch):
        for x in range(pitch // 2, width - pitch // 2, pitch):
            cy, cx = y + rng.integers(-jitter, jitter + 1), x + rng.integers(-jitter, jitter + 1)
            image[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1] = 0
            centres.append((cy, cx))
    return image, np.array(centres, dtype=float)


def _resize_like_preprocessing(patch: np.ndarray, patch_size: int) -> np.ndarray:
    """Same resizing as queue_worker in patch_extraction.py"""
    if patch.shape[0] == patch_size:
        return patch
    tile = Image.fromarray(patch)
    resample = getattr(Image, "Resampling", Image).LANCZOS
    if patch.shape[0] > patch_size:
        tile.thumbnail((patch_size, patch_size), resample)
    else:
        tile = tile.resize((patch_size, patch_size), resample)
    return np.array(tile, dtype=np.uint8)


@pytest.mark.parametrize("target_mpp", [0.25, 0.5, 0.27, 0.6, 0.125])
def test_grid_maps_patch_pixels_back_to_the_slide(target_mpp):
    """Extract patches exactly like the preprocessing and map the dots found in them back to the slide"""
    pytest.importorskip("openslide")
    from openslide import ImageSlide
    from scipy import ndimage
    from scipy.spatial import cKDTree

    from preprocessing.patch_extraction.src.utils.patch_util import (
        DeepZoomGeneratorOS,
        pad_tile,
        target_mpp_to_downsample,
    )

    base_mpp, patch_size, overlap = 0.25, 256, 16  # overlap per side, as in the preprocessing
    image, centres = _dot_slide()
    tree = cKDTree(centres)

    downsample, rescaling = target_mpp_to_downsample(base_mpp, target_mpp)
    tile_size, tile_overlap = patch_to_tile_size(patch_size, overlap, rescaling)
    tiles = DeepZoomGeneratorOS(
        ImageSlide(Image.fromarray(image)), tile_size=tile_size, overlap=tile_overlap, limit_bounds=True
    )
    level = tiles.level_count - downsample.bit_length()
    grid = PatchGrid.from_metadata(
        {
            "patch_size": patch_size,
            "patch_overlap": 2 * overlap,
            "downsampling": downsample,
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
        }
    )
    assert grid.scale == pytest.approx(target_mpp / base_mpp, rel=0.03)

    errors, former_errors = [], []
    n_cols, n_rows = tiles.level_tiles[level]
    for row in range(n_rows):
        for col in range(n_cols):
            tile = np.array(tiles.get_tile(level, (col, row)), dtype=np.uint8)
            patch = pad_tile(tile, tile_size + 2 * tile_overlap, col, row)
            patch = _resize_like_preprocessing(patch, patch_size)
            darkness = 255.0 - patch.mean(axis=-1)
            labels, n_dots = ndimage.label(darkness > 64)
            if n_dots == 0:
                continue
            ids = np.arange(1, n_dots + 1)
            coms = np.array(ndimage.center_of_mass(darkness, labels, ids))
            for dot_slice, com in zip(ndimage.find_objects(labels), coms):
                if any(s.start == 0 or s.stop == patch_size for s in dot_slice):
                    continue  # dot cut by the patch border
                # pixel centres: patch pixel i spans level-0 [o + i * scale, o + (i + 1) * scale)
                mapped = grid.to_level0(com + 0.5, row, col) - 0.5
                errors.append(tree.query(mapped)[0])
                former_errors.append(
                    tree.query(_former_origin(row, col, patch_size, 2 * overlap, downsample) + com)[0]
                )

    errors = np.array(errors)
    assert len(errors) > 100
    assert errors.max() < 1.5, f"max error {errors.max():.2f} px"
    assert errors.mean() < 0.5
    former_errors = np.array(former_errors)
    if grid.scale == 1:
        assert np.allclose(former_errors, errors)  # the former formula was right only here
    else:
        assert np.median(former_errors) > 5  # and the test detects its error otherwise
