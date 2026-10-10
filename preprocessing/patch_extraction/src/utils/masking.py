# -*- coding: utf-8 -*-
# Masking function to generate tissue masks
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from typing import List, Tuple, Union

import cv2
import numpy as np
import skimage.color as sk_color
import skimage.filters as sk_filters
import skimage.morphology as sk_morphology
from PIL import Image
from rasterio.features import rasterize
from shapely.affinity import scale
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from preprocessing.patch_extraction import logger


def as_geometry_list(polygons: Union[BaseGeometry, List[BaseGeometry]]) -> List[BaseGeometry]:
    """Accept a single geometry or an iterable of geometries and always return a list."""
    if isinstance(polygons, BaseGeometry):
        return [polygons]
    return list(polygons)


def rasterize_polygons(
    polygons: Union[BaseGeometry, List[BaseGeometry]],
    shape: Tuple[int, int],
    value: int = 1,
) -> np.ndarray:
    """Rasterize polygons given in pixel coordinates into a uint8 mask.

    Pixels whose center lies inside a polygon (holes excluded) get `value`, all others 0.
    This replaces the previous approach (temporary GeoTIFF on disk + rasterio.mask), which
    was not safe for parallel runs; the semantics are identical.

    Args:
        polygons (Union[BaseGeometry, List[BaseGeometry]]): Polygon(s) in pixel coordinates of the mask.
        shape (Tuple[int, int]): Output shape (height, width).
        value (int, optional): Value assigned inside the polygons. Defaults to 1.

    Returns:
        np.ndarray: Mask with shape (height, width) and dtype uint8
    """
    polygons = [p for p in as_geometry_list(polygons) if not p.is_empty]
    out = np.zeros(tuple(shape), dtype=np.uint8)
    if len(polygons) == 0:
        return out
    return rasterize(((p, value) for p in polygons), out=out, fill=0)


def generate_tissue_mask(
    tissue_tile: np.ndarray,
    mask_otsu: bool = False,
    polygons: List[Polygon] = None,
    region_labels: List[str] = None,
    otsu_annotation: Union[List[str], str] = "object",
    downsample: int = 1,
    apply_prefilter: bool = False,
) -> np.ndarray:
    """Generate a tissue mask using otsu thresholding.

    Per Default, otsu-thresholding is performed. If mask_otsu is true, first a masked image is calculate
    using the annotation matching the otsu_annotation label.

    Args:
        tissue_tile (np.ndarray): Tissue tile as numpy array with shape (height, width, 3)
        mask_otsu (bool, optional): If masking is applied before thresholding. Defaults to False.
        polygons (List[Polygon], optional):  Annotations of this WSI as a list of polygons (referenced to highest level of WSI). Defaults to None.
        region_labels (List[str], optional): List of labels for the annotations provided as polygons parameter. Defaults to None.
        otsu_annotation (Union[List[str], str], optional):  List with annotation names or string with annotation name to use for a masked otsu thresholding.
            Multiple names in one string can be separated by whitespace. Defaults to "object".
        downsample (int, optional): Downsampling of the tissue tile compared to highest WSI level. Used for matching annotations with tissue-tile size.
            Defaults to 1.
        apply_prefilter (bool, optional): If a prefilter should be used to remove markers before applying otsu. Defaults to False.

    Returns:
        np.ndarray: Binary tissue mask with shape (height, width). 1 = background, 0 = tissue
    """
    if polygons is not None:
        assert len(polygons) == len(
            region_labels
        ), "Polygon list and polygon labels are not having the same length"

    if mask_otsu:
        # filter
        otsu_polgyon = get_filtered_polygons(
            polygons=polygons or [],
            region_labels=region_labels or [],
            filter_labels=otsu_annotation,
            downsample=downsample,
        )
        if len(otsu_polgyon) != 0:
            logger.debug(
                "Mask tissue thumbnail with region before applying Otsu thresholding"
            )
            tissue_tile = mask_tile_with_region(tile=tissue_tile, polygons=otsu_polgyon)
        else:
            logger.error(
                "Annotation with given label does not exist. Using unmasked thresholding"
            )
    # apply otsu thresholding

    if apply_prefilter:
        tissue_tile = remove_marker_filters(tile=tissue_tile)

    tissue_mask = apply_otsu_thresholding(tile=tissue_tile)
    assert len(np.unique(tissue_mask)) <= 2, "Mask is not binary"

    return tissue_mask


def convert_polygons_to_mask(
    polygons: Union[List[Polygon], Polygon],
    reference_size: Tuple[int],
    downsample: int = 1,
) -> np.ndarray:
    """Convert a polygon to a mask

    The function is assuming that polygons have already been filtered (see get_filtered_polygon).

    Args:
        polygons (Union[List[Polygon], Polygon]): List of polygons converted to a mask. Can work with Polygons with holes inside
        reference_size (tuple[int]): Shape of resulting mask image. Shape should be (height, width, channels).
        downsample (int, optional): Set the factor by which the polygon should be scaled down. Defaults to 1.

    Returns:
        np.ndarray: Binary mask with shape (height, width). 1 = background, 0 = inside the polygons (tissue)
    """
    polygons_downsampled = [
        scale(
            poly,
            xfact=1 / downsample,
            yfact=1 / downsample,
            origin=(0, 0),
        )
        for poly in as_geometry_list(polygons)
    ]
    inside = rasterize_polygons(polygons_downsampled, shape=tuple(reference_size[:2]), value=1)
    mask = (1 - inside).astype(np.uint8)

    assert len(np.unique(mask)) <= 2, "Mask is not binary"

    return mask


def get_filtered_polygons(
    polygons: List[Polygon],
    region_labels: List[str],
    filter_labels: Union[List[str], str],
    downsample: int = 1,
) -> List[Polygon]:
    """Filter Polygons by a list of filter labels

    Returns a list with filtered polygons containing just the polygons with
    the label specified in filter_labels

    Args:
        polygons (List[Polygon]): Annotations as a list of polygons.
        region_labels (List[str]): List of labels
        filter_labels (Union[List[str], str]): Labels to keep. A single string may contain several
            labels separated by whitespace.
        downsample (int, optional): Scaling factor to downscale polygon. Defaults to 1.

    Returns:
        List[Polygon]: List with filtered polygons
    """
    if isinstance(filter_labels, str):
        filter_labels = filter_labels.split()
    filter_labels = set(filter_labels)
    logger.debug(
        f"Filter polygons for label: {filter_labels} and downsample results to {downsample}"
    )
    filtered_polygons = []
    for poly, region_label in zip(polygons, region_labels):
        if region_label in filter_labels:
            filtered_polygons.append(
                scale(poly, xfact=1 / downsample, yfact=1 / downsample, origin=(0, 0))
            )
    if len(filtered_polygons) == 0:
        logger.debug(
            "ValueError: Annotation with given label does not exist or Annotation has a non-valid Type."
        )

    return filtered_polygons


def mask_tile_with_region(
    tile: np.ndarray, polygons: Union[List[Polygon], Polygon]
) -> np.ndarray:
    """Mask a tile with a region and return the masked tile (everything outside the region becomes white)

    Args:
        tile (np.ndarray): Tile which should be masked
        polygons (Union[List[Polygon], Polygon]): List of mask polygons or a polygon to mask

    Returns:
        np.ndarray: Masked tile
    """
    mask = rasterize_polygons(polygons, shape=tile.shape[:2], value=255)
    # create masked figure
    fg = cv2.bitwise_or(tile, tile, mask=mask)
    inverse_mask = cv2.bitwise_not(mask)
    background = np.full(tile.shape, 255, dtype=np.uint8)
    bk = cv2.bitwise_or(background, background, mask=inverse_mask)

    return cv2.bitwise_or(fg, bk)


def apply_otsu_thresholding(tile: np.ndarray) -> np.ndarray:
    """Generate a binary tissue mask by using Otsu thresholding

    Args:
        tile (np.ndarray): Tile with tissue with shape (height, width, 3)

    Returns:
        np.ndarray: Binary mask with shape (height, width). 1 = background, 0 = tissue
    """
    hsv_img = cv2.cvtColor(tile.astype(np.uint8), cv2.COLOR_RGB2HSV)
    gray_mask = cv2.inRange(hsv_img, (0, 0, 70), (180, 10, 255))
    black_mask = cv2.inRange(hsv_img, (0, 0, 0), (180, 255, 85))
    # Set all grey/black pixels to white
    full_tile_bg = np.copy(tile)
    full_tile_bg[np.where(gray_mask | black_mask)] = 255

    # apply otsu mask first time for removing larger artifacts
    masked_image_gray = 255 * sk_color.rgb2gray(full_tile_bg)
    thresh = sk_filters.threshold_otsu(masked_image_gray)
    otsu_masking = masked_image_gray < thresh
    # improving mask
    otsu_masking = sk_morphology.remove_small_objects(otsu_masking, 60)
    otsu_masking = sk_morphology.dilation(otsu_masking, sk_morphology.square(12))
    otsu_masking = sk_morphology.closing(otsu_masking, sk_morphology.square(5))
    otsu_masking = sk_morphology.remove_small_holes(otsu_masking, 250)
    tile = mask_rgb(tile, otsu_masking).astype(np.uint8)

    # apply otsu mask second time for removing small artifacts
    masked_image_gray = 255 * sk_color.rgb2gray(tile)
    thresh = sk_filters.threshold_otsu(masked_image_gray)
    otsu_masking = masked_image_gray < thresh
    otsu_masking = sk_morphology.remove_small_holes(otsu_masking, 5000)
    otsu_thr = ~otsu_masking
    otsu_thr = otsu_thr.astype(np.uint8)

    return otsu_thr


def mask_rgb(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask an RGB image

    Args:
        rgb (np.ndarray): RGB image to mask with shape (height, width, 3)
        mask (np.ndarray): Binary mask with shape (height, width)

    Returns:
        np.ndarray: Masked image
    """
    assert (
        rgb.shape[:-1] == mask.shape
    ), "Mask and RGB shape are different. Cannot mask when source and mask have different dimension."
    mask_positive = np.dstack([mask, mask, mask])
    mask_negative = np.dstack([~mask, ~mask, ~mask])
    positive = rgb * mask_positive
    negative = rgb * mask_negative
    negative = 255 * (negative > 0.0001).astype(int)

    masked_image = positive + negative

    return np.clip(masked_image, a_min=0, a_max=255)


def remove_marker_filters(tile: np.ndarray) -> np.ndarray:
    """Remove pen markers (red, green, blue) from a tile; removed pixels become white

    Args:
        tile (np.ndarray): Tile with tissue with shape (height, width, 3)

    Returns:
        np.ndarray: Filtered tile with shape (height, width, 3)
    """
    # histolab is only needed for the optional marker pre-filter: import lazily
    from histolab.filters.image_filters import BluePenFilter, GreenPenFilter, RedPenFilter

    red_pen_filter = RedPenFilter()
    green_pen_filter = GreenPenFilter()
    blue_pen_filter = BluePenFilter()

    tile = Image.fromarray(tile.astype(np.uint8))

    tile = blue_pen_filter(tile)
    tile = green_pen_filter(tile)
    tile = red_pen_filter(tile)

    image_rgb_np = np.array(tile)
    black_pixels = (
        (image_rgb_np[:, :, 0] == 0)
        & (image_rgb_np[:, :, 1] == 0)
        & (image_rgb_np[:, :, 2] == 0)
    )
    image_rgb_np[black_pixels] = [255, 255, 255]

    return image_rgb_np
