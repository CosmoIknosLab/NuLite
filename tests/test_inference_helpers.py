# -*- coding: utf-8 -*-
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from shapely import strtree
from shapely.geometry import Polygon

from nuclei_detection.inference.nuclei_detection import (
    CellPostProcessor,
    CellSegmentationInference,
    _fix_polygon,
    _strtree_query,
    check_wsi,
    get_cell_position,
    get_cell_position_marging,
    get_edge_patch,
)

PATCH_SIZE = 1024
OVERLAP = 64


def test_get_edge_patch_covers_all_border_combinations():
    for top in (0, 1):
        for right in (0, 1):
            for down in (0, 1):
                for left in (0, 1):
                    position = [top, right, down, left]
                    neighbours = get_edge_patch(position, 5, 7)
                    if any(position):
                        assert len(neighbours) > 0, position
                        assert [5, 7] not in neighbours
                    else:
                        assert neighbours == []
    assert get_edge_patch([1, 0, 0, 0], 5, 7) == [[4, 7]]
    assert get_edge_patch([1, 1, 0, 0], 5, 7) == [[4, 7], [5, 8], [4, 8]]
    assert get_edge_patch([0, 0, 1, 1], 5, 7) == [[6, 7], [5, 6], [6, 6]]
    # cells spanning two opposite borders previously returned None
    assert get_edge_patch([1, 0, 1, 0], 5, 7) == [[4, 7], [6, 7]]


def test_cell_position_and_margin_status():
    assert get_cell_position(np.array([[0, 10], [20, 30]]), PATCH_SIZE) == [1, 0, 0, 0]
    assert get_cell_position(np.array([[10, 1000], [20, 1024]]), PATCH_SIZE) == [0, 1, 0, 0]
    assert get_cell_position_marging(np.array([[500, 500], [520, 530]]), PATCH_SIZE, OVERLAP) == 0
    assert get_cell_position_marging(np.array([[10, 500], [30, 530]]), PATCH_SIZE, OVERLAP) == 2
    assert get_cell_position_marging(np.array([[10, 10], [30, 30]]), PATCH_SIZE, OVERLAP) == 1
    assert get_cell_position_marging(np.array([[1000, 1000], [1020, 1020]]), PATCH_SIZE, OVERLAP) == 5
    assert get_cell_position_marging(np.array([[500, 10], [520, 30]]), PATCH_SIZE, OVERLAP) == 8
    rng = np.random.default_rng(0)
    for _ in range(500):
        r0, c0 = rng.integers(0, PATCH_SIZE - 1, 2)
        r1, c1 = rng.integers(r0 + 1, PATCH_SIZE + 1), rng.integers(c0 + 1, PATCH_SIZE + 1)
        assert get_cell_position_marging(np.array([[r0, c0], [r1, c1]]), PATCH_SIZE, OVERLAP) is not None


def test_fix_polygon_keeps_largest_part():
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    assert not bowtie.is_valid
    fixed = _fix_polygon(bowtie)
    assert isinstance(fixed, Polygon)
    assert fixed.is_valid
    assert abs(fixed.area - 1.0) < 1e-9
    valid = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert _fix_polygon(valid).equals(valid)


def test_strtree_query_returns_indices():
    squares = [Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i in (0, 5, 10)]
    tree = strtree.STRtree(squares)
    hits = _strtree_query(tree, squares[1])
    assert all(isinstance(h, int) for h in hits)
    assert 1 in hits
    assert 0 not in hits and 2 not in hits


def _global_offset(row, col):
    x_global = int(row * PATCH_SIZE - (row + 0.5) * OVERLAP)
    y_global = int(col * PATCH_SIZE - (col + 0.5) * OVERLAP)
    return np.array([x_global, y_global])


def _make_cell(row, col, bbox, cell_type=1):
    """Build a cell dict like CellSegmentationInference.process_wsi does (bbox in patch pixels, [[rmin, cmin], [rmax, cmax]])"""
    bbox = np.array(bbox)
    offset = _global_offset(row, col)
    (rmin, cmin), (rmax, cmax) = bbox + offset
    contour = np.array([[cmin, rmin], [cmax, rmin], [cmax, rmax], [cmin, rmax]])  # x, y
    centroid = np.array([(cmin + cmax) / 2, (rmin + rmax) / 2])
    cell = {
        "bbox": (bbox + offset).tolist(),
        "centroid": centroid.tolist(),
        "contour": contour.tolist(),
        "type_prob": 0.9,
        "type": cell_type,
        "patch_coordinates": [row, col],
        "cell_status": get_cell_position_marging(bbox, PATCH_SIZE, OVERLAP),
        "offset_global": offset.tolist(),
    }
    if np.max(bbox) == PATCH_SIZE or np.min(bbox) == 0:
        position = get_cell_position(bbox, PATCH_SIZE)
        cell["edge_position"] = True
        cell["edge_information"] = {"position": position, "edge_patches": get_edge_patch(position, row, col)}
    else:
        cell["edge_position"] = False
    return cell


def test_cell_post_processor_removes_border_and_duplicated_cells():
    cells = [
        # mid cells, always kept
        _make_cell(0, 0, [[300, 300], [320, 325]]),
        _make_cell(0, 1, [[300, 300], [320, 325]]),
        # a cell cut by the right border of patch (0,0): neighbour patch (0,1) exists -> dropped ...
        _make_cell(0, 0, [[500, 1000], [520, 1024]]),
        # ... the same cell seen completely in the left margin of patch (0,1) -> kept
        _make_cell(0, 1, [[500, 40], [520, 64]]),
        # the same (non border) cell detected in the right margin of (0,0) and in the left margin of (0,1)
        # (global cols 933..958 lie in the 64 px overlap zone 928..992 of the two patches):
        # both are margin cells and overlap in global coordinates -> only the bigger one survives
        _make_cell(0, 0, [[600, 965], [620, 990]]),
        _make_cell(0, 1, [[600, 5], [621, 31]]),
        # a border cell whose neighbour patch was never processed -> kept
        _make_cell(0, 1, [[0, 300], [20, 330]]),
    ]
    # sanity: the duplicated cells really share the same global polygon
    assert Polygon(cells[2]["contour"]).equals(Polygon(cells[3]["contour"]))
    assert Polygon(cells[4]["contour"]).intersection(Polygon(cells[5]["contour"])).area > 0

    processor = CellPostProcessor(cells, logging.getLogger("test"))
    kept = sorted(processor.post_process_cells().index.values)
    assert kept == [0, 1, 3, 5, 6]


def test_convert_geojson_does_not_mutate_contours():
    cells = [
        {"type": 1, "centroid": [1.0, 2.0], "contour": [[0, 0], [1, 0], [1, 1]]},
        {"type": 2, "centroid": [3.0, 4.0], "contour": [[0, 0], [2, 0], [2, 2]]},
    ]
    geojson = CellSegmentationInference.convert_geojson(None, cells, polygons=True)
    assert len(geojson) == 2
    assert all(len(c["contour"]) == 3 for c in cells)
    ring = geojson[0]["geometry"]["coordinates"][0][0]
    assert ring[0] == ring[-1]  # closed ring
    points = CellSegmentationInference.convert_geojson(None, cells, polygons=False)
    assert points[0]["geometry"]["coordinates"] == [[1.0, 2.0]]
    # a WSI without any detected cell must not crash the export
    assert CellSegmentationInference.convert_geojson(None, [], polygons=True) == []
    assert CellSegmentationInference.convert_geojson(None, [], polygons=False) == []


def test_check_wsi_validates_grid_parameters():
    good = SimpleNamespace(metadata={"magnification": 40.0, "base_magnification": 40.0, "downsampling": 1, "patch_size": 1024, "patch_overlap": 64})
    check_wsi(good, magnification=40)
    with pytest.raises(RuntimeError):
        check_wsi(SimpleNamespace(metadata={**good.metadata, "patch_overlap": 32}), magnification=40)
    with pytest.raises(RuntimeError):
        check_wsi(SimpleNamespace(metadata={**good.metadata, "patch_size": 512}), magnification=40)
    with pytest.raises(RuntimeError):
        check_wsi(good, magnification=20)
