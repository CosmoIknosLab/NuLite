# -*- coding: utf-8 -*-
# Train NuLite with a yaml configuration and evaluate the selected checkpoint on the test folds
import inspect
import os
import sys

# make the repository root importable when the script is started from another working directory
_repo_root = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from base_ml.base_cli import ExperimentBaseParser  # noqa: E402
from nuclei_detection.experiments.experiment_nulite_pannuke import ExperimentNuLitePanNuke  # noqa: E402
from nuclei_detection.inference.inference_nufastvit_experiment_pannuke import InferenceNuLite  # noqa: E402

EXPERIMENTS = {"pannuke": ExperimentNuLitePanNuke}


if __name__ == "__main__":
    # Parse arguments
    configuration_parser = ExperimentBaseParser()
    configuration = configuration_parser.parse_arguments()

    dataset_name = str(configuration["data"]["dataset"]).lower()
    if dataset_name not in EXPERIMENTS:
        raise NotImplementedError(
            f"Unknown dataset '{dataset_name}'. Implemented experiments: {list(EXPERIMENTS)}"
        )
    experiment_class = EXPERIMENTS[dataset_name]

    # Setup and run experiment (optionally resuming from a checkpoint passed via --checkpoint)
    experiment = experiment_class(
        default_conf=configuration, checkpoint=configuration.get("checkpoint")
    )
    outdir = experiment.run_experiment()

    # Evaluate on the test folds
    inference = InferenceNuLite(
        run_dir=outdir,
        gpu=configuration["gpu"],
        checkpoint_name=configuration.get("eval_checkpoint", "model_best.pth"),
        magnification=configuration["data"].get("magnification", 40),
        reparameterize=configuration["model"].get("reparameterize", True),
    )
    (
        trained_model,
        inference_dataloader,
        dataset_config,
    ) = inference.setup_patch_inference()
    inference.run_patch_inference(
        trained_model,
        inference_dataloader,
        dataset_config,
        generate_plots=False,
    )
