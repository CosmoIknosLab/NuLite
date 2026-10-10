# -*- coding: utf-8 -*-
import torch

from utils.tools import AverageMeter, DataclassHVStorage, flatten_dict, unflatten_dict


def test_flatten_unflatten_roundtrip():
    conf = {
        "model": {"backbone": "fastvit_t8"},
        "data": {"num_nuclei_classes": 6, "folds": [0, 1]},
        "dataset_config": {"tissue_types": {"Bile-duct": 1, "Adrenal_gland": 0}},
    }
    flat = flatten_dict(conf, sep=".")
    assert flat["dataset_config.tissue_types.Bile-duct"] == 1
    assert flat["data.folds"] == [0, 1]
    assert unflatten_dict(flat, ".") == conf


def test_get_dict_does_not_mutate_storage():
    storage = DataclassHVStorage(
        nuclei_binary_map=torch.zeros(1),
        hv_map=torch.zeros(1),
        tissue_types=torch.zeros(1),
        nuclei_type_map=torch.zeros(1),
        instance_map=None,
        instance_types_nuclei=None,
        batch_size=1,
        regression_map=torch.zeros(1),
    )
    as_dict = storage.get_dict()
    assert "regression_map" not in as_dict  # regression_loss is False
    assert storage.regression_map is not None  # the dataclass itself is untouched
    as_dict["hv_map"] = None
    assert storage.hv_map is not None
    # a second call works as well
    assert "regression_map" not in storage.get_dict()


def test_average_meter_format():
    meter = AverageMeter("acc", ":.4f")
    meter.update(0.5)
    meter.update(1.0)
    assert abs(meter.avg - 0.75) < 1e-9
    assert str(meter) == "acc 1.0000 (0.7500)"
