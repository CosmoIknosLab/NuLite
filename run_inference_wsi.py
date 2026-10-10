# -*- coding: utf-8 -*-
# Entry point for NuLite inference on preprocessed (patched) WSIs
from pathlib import Path
from typing import List

from nuclei_detection.datamodel.wsi_datamodel import WSI
from nuclei_detection.inference.nuclei_detection import (
    CellSegmentationInference,
    InferenceWSIParser,
    check_wsi,
)
from utils.file_handling import load_wsi_files_from_csv


def run_single_wsi(
    cell_segmentation: CellSegmentationInference,
    wsi_path: Path,
    patched_slide_path: Path,
    configuration: dict,
) -> None:
    """Check and process one WSI with the settings from the CLI."""
    wsi_name = wsi_path.stem
    wsi_file = WSI(
        name=wsi_name,
        patient=wsi_name,
        slide_path=wsi_path,
        patched_slide_path=patched_slide_path,
    )
    check_wsi(wsi=wsi_file, magnification=configuration["magnification"])
    cell_segmentation.process_wsi(
        wsi_file,
        subdir_name=configuration["outdir_subdir"],
        geojson=configuration["geojson"],
        batch_size=configuration["batch_size"],
    )


def collect_wsi_files(configuration: dict) -> List[Path]:
    """Collect the WSI files of a dataset, either from a csv filelist or by globbing the folder."""
    wsi_root = Path(configuration["wsi_paths"])
    if configuration["filelist"] is not None:
        if Path(configuration["filelist"]).suffix != ".csv":
            raise ValueError("Filelist must be a .csv file!")
        wsi_filelist = load_wsi_files_from_csv(
            csv_path=configuration["filelist"],
            wsi_extension=configuration["wsi_extension"],
        )
        # entries may be absolute paths or paths relative to wsi_paths
        return [Path(f) if Path(f).is_absolute() else wsi_root / f for f in wsi_filelist]
    return sorted(wsi_root.glob(f"**/*.{configuration['wsi_extension']}"))


if __name__ == "__main__":
    configuration_parser = InferenceWSIParser()
    configuration = configuration_parser.parse_arguments()
    command = configuration["command"]
    if command is None:
        configuration_parser.parser.error("Please select a subcommand: process_wsi or process_dataset")

    cell_segmentation = CellSegmentationInference(
        model_path=configuration["model"],
        gpu=configuration["gpu"],
    )

    if command.lower() == "process_wsi":
        cell_segmentation.logger.info("Processing single WSI file")
        run_single_wsi(
            cell_segmentation,
            wsi_path=Path(configuration["wsi_path"]),
            patched_slide_path=Path(configuration["patched_slide_path"]),
            configuration=configuration,
        )

    elif command.lower() == "process_dataset":
        cell_segmentation.logger.info("Processing whole dataset")
        if configuration["filelist"] is not None:
            cell_segmentation.logger.info(
                f"Loading files from filelist {configuration['filelist']}"
            )
        else:
            cell_segmentation.logger.info(
                f"Loading all files from folder {configuration['wsi_paths']}. No filelist provided."
            )
        wsi_filelist = collect_wsi_files(configuration)
        for i, wsi_path in enumerate(wsi_filelist):
            wsi_path = Path(wsi_path)
            cell_segmentation.logger.info(f"File {i + 1}/{len(wsi_filelist)}: {wsi_path.stem}")
            run_single_wsi(
                cell_segmentation,
                wsi_path=wsi_path,
                patched_slide_path=Path(configuration["patch_dataset_path"]) / wsi_path.stem,
                configuration=configuration,
            )
