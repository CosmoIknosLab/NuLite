# -*- coding: utf-8 -*-
import warnings

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import Polygon, box

from preprocessing.patch_extraction.src.cli import PreProcessingConfig
from preprocessing.patch_extraction.src.utils.masking import (
    convert_polygons_to_mask,
    get_filtered_polygons,
    mask_tile_with_region,
    rasterize_polygons,
)
from preprocessing.patch_extraction.src.utils.patch_util import (
    background_ratio_from_mask,
    compute_overlap,
    compute_patch_location_in_level,
    get_files_from_dir,
    is_power_of_two,
    patch_to_tile_size,
)


def test_background_ratio_is_robust_to_unsigned_overflow():
    # thumbnail tile with overlap (256x256) is larger than the reference patch area (240x240):
    # with numpy >= 2 the former uint64 arithmetic wrapped around and rejected every tissue patch
    tissue = np.zeros((256, 256), dtype=np.uint8)  # 0 = tissue
    assert background_ratio_from_mask(tissue, 240) == 0.0
    background = np.ones((256, 256), dtype=np.uint8)
    assert background_ratio_from_mask(background, 240) == 1.0
    half = np.zeros((240, 240), dtype=np.uint8)
    half[:120] = 1
    assert abs(background_ratio_from_mask(half, 240) - 0.5) < 1e-9
    # tile smaller than the patch (slide border): the missing pixels count as background
    cropped = np.zeros((200, 240), dtype=np.uint8)
    assert abs(background_ratio_from_mask(cropped, 240) - (40 * 240) / 240**2) < 1e-9
    assert background_ratio_from_mask(np.zeros((240, 240), dtype=bool), 240) == 0.0


def test_config_validators_reject_out_of_range_values(tmp_path):
    with pytest.raises(Exception):
        PreProcessingConfig(output_path=str(tmp_path), wsi_paths=str(tmp_path), patch_overlap=150)
    with pytest.raises(Exception):
        PreProcessingConfig(output_path=str(tmp_path), wsi_paths=str(tmp_path), min_intersection_ratio=2.0)
    with pytest.raises(Exception):
        PreProcessingConfig(output_path=str(tmp_path), wsi_paths=str(tmp_path), patch_size=0)
    with pytest.raises(RuntimeError):
        PreProcessingConfig(output_path=str(tmp_path))  # neither wsi_paths nor wsi_filelist


def test_config_defaults(tmp_path):
    cfg = PreProcessingConfig(output_path=str(tmp_path), wsi_paths=str(tmp_path), patch_overlap=6.25, log_level="info")
    assert cfg.log_level == "INFO"
    assert cfg.label_map == {"background": 0}
    assert cfg.tissue_annotation_intersection_ratio == cfg.min_intersection_ratio
    assert cfg.output_path == tmp_path.resolve()
    as_dict = cfg.to_dict()
    assert as_dict["patch_size"] == 256


def _reference_mask(polygons, shape, tmp_path):
    """The former implementation: white GeoTIFF on disk + rasterio.mask"""
    Image.fromarray(255 * np.ones(shape=shape, dtype=np.uint8)).save(tmp_path / "tmp.tif")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with rasterio.open(tmp_path / "tmp.tif") as src:
            out_image, _ = rasterio_mask(src, polygons, crop=False)
    return (out_image[0] > 0).astype(np.uint8)


def test_rasterize_polygons_matches_former_rasterio_mask(tmp_path):
    rng = np.random.default_rng(0)
    shape = (120, 160)
    for _ in range(5):
        polygons = []
        for _ in range(rng.integers(1, 4)):
            cx, cy = rng.uniform(10, 150), rng.uniform(10, 110)
            pts = [(cx + rng.uniform(5, 30) * np.cos(a), cy + rng.uniform(5, 30) * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 9)[:-1]]
            polygons.append(Polygon(pts).buffer(0))
        # polygon with a hole
        polygons.append(box(60, 40, 100, 80).difference(box(70, 50, 80, 60)))
        ours = rasterize_polygons(polygons, shape=shape, value=1)
        reference = _reference_mask(polygons, shape, tmp_path)
        assert ours.shape == shape
        assert np.array_equal(ours, reference)


def test_convert_polygons_to_mask_convention():
    square = box(10, 10, 20, 20)
    mask = convert_polygons_to_mask(square, reference_size=(40, 40, 3), downsample=1)
    assert mask.shape == (40, 40)
    assert mask[15, 15] == 0  # inside -> tissue
    assert mask[5, 5] == 1  # outside -> background
    mask_ds = convert_polygons_to_mask([square], reference_size=(20, 20, 3), downsample=2)
    assert mask_ds[7, 7] == 0 and mask_ds[2, 2] == 1


def test_mask_tile_with_region_whitens_outside():
    tile = np.full((30, 30, 3), 100, dtype=np.uint8)
    masked = mask_tile_with_region(tile, box(5, 5, 15, 15))
    assert (masked[10, 10] == 100).all()
    assert (masked[25, 25] == 255).all()


def test_get_filtered_polygons_accepts_whitespace_separated_labels():
    polys = [box(0, 0, 1, 1), box(2, 2, 3, 3), box(4, 4, 5, 5)]
    labels = ["tumor", "stroma", "tumor"]
    assert len(get_filtered_polygons(polys, labels, "tumor")) == 2
    assert len(get_filtered_polygons(polys, labels, "tumor stroma")) == 3
    assert len(get_filtered_polygons(polys, labels, ["stroma"])) == 1
    # no accidental substring matching
    assert len(get_filtered_polygons(polys, labels, "tum")) == 0


def test_grid_helpers():
    assert patch_to_tile_size(1024, 32) == (960, 32)
    assert is_power_of_two(8) and not is_power_of_two(6)
    # (cols, rows) grid of 3x3 with overlap 32
    assert compute_overlap(0, 0, (3, 3), 32) == (0, 32, 32, 0)  # top, right, bottom, left
    assert compute_overlap(1, 1, (3, 3), 32) == (32, 32, 32, 32)
    assert compute_overlap(2, 2, (3, 3), 32) == (32, 0, 0, 32)
    assert compute_patch_location_in_level(0, 0, 100, (3, 3), 10) == (0, 110, 0, 110)
    assert compute_patch_location_in_level(1, 1, 100, (3, 3), 10) == (90, 210, 90, 210)


def test_get_files_from_dir(tmp_path):
    (tmp_path / "a.svs").touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.svs").touch()
    (tmp_path / "sub" / "not_svs.txt").touch()
    assert [f.name for f in get_files_from_dir(tmp_path, "svs")] == ["a.svs"]
    nested = tmp_path / "nested"
    (nested / "class1").mkdir(parents=True)
    (nested / "class1" / "c.svs").touch()
    (nested / "class1" / "c_svs").touch()  # must not match "**/*svs"
    assert [f.name for f in get_files_from_dir(nested, "svs")] == ["c.svs"]
