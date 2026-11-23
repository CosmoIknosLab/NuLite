# -*- coding: utf-8 -*-
from pathlib import Path

import pytest

pe = pytest.importorskip("preprocessing.patch_extraction.src.patch_extraction")


class FakeOpenSlide:
    def __init__(self, path):
        self.path = path


class FakeCuImage:
    def __init__(self, path):
        if "broken" in path:
            raise RuntimeError(
                "This format has more than one image with Subfile Type 0 so cannot be loaded!"
            )
        self.path = path


class FakeCucimDeepZoom:
    pass


def _processor(image_loader, deepzoomgenerator):
    processor = object.__new__(pe.PreProcessor)  # no files/config needed for _open_slide
    processor.image_loader = image_loader
    processor.deepzoomgenerator = deepzoomgenerator
    processor._cucim_fallback_files = set()
    return processor


def test_open_slide_falls_back_to_openslide_when_cucim_fails(monkeypatch):
    monkeypatch.setattr(pe, "OpenSlide", FakeOpenSlide)
    processor = _processor(FakeCuImage, FakeCucimDeepZoom)

    slide, slide_cu, deepzoom = processor._open_slide(Path("ok.svs"))
    assert isinstance(slide, FakeOpenSlide) and isinstance(slide_cu, FakeCuImage)
    assert deepzoom is FakeCucimDeepZoom

    # the file CuCIM cannot open is read with OpenSlide instead of stopping the preprocessing
    for _ in range(2):
        slide, slide_cu, deepzoom = processor._open_slide(Path("broken.svs"))
        assert isinstance(slide, FakeOpenSlide) and slide_cu is slide
        assert deepzoom is pe.DeepZoomGeneratorOS
    assert processor._cucim_fallback_files == {"broken.svs"}  # warned once per file
    # the other files keep using CuCIM
    assert processor._open_slide(Path("ok.svs"))[2] is FakeCucimDeepZoom


def test_open_slide_with_openslide_selected(monkeypatch):
    monkeypatch.setattr(pe, "OpenSlide", FakeOpenSlide)
    processor = _processor(FakeOpenSlide, pe.DeepZoomGeneratorOS)
    slide, slide_cu, deepzoom = processor._open_slide(Path("broken.svs"))
    assert slide_cu is slide and deepzoom is pe.DeepZoomGeneratorOS
    assert processor._cucim_fallback_files == set()
