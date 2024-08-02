from nuclei_detection.datamodel.wsi_datamodel import WSI
from nuclei_detection.inference.nuclei_detection import check_wsi, InferenceWSIParser, CellSegmentationInference
from utils.file_handling import load_wsi_files_from_csv
from pathlib import Path

if __name__ == '__main__':
    configuration_parser = InferenceWSIParser()
    configuration = configuration_parser.parse_arguments()
    command = configuration["command"]
    cell_segmentation = CellSegmentationInference(
        model_path=configuration["model"],
        gpu=configuration["gpu"],
    )

    if command.lower() == "process_wsi":
        cell_segmentation.logger.info("Processing single WSI file")
        wsi_path = Path(configuration["wsi_path"])
        wsi_name = wsi_path.stem
        wsi_file = WSI(
            name=wsi_name,
            patient=wsi_name,
            slide_path=wsi_path,
            patched_slide_path=configuration["patched_slide_path"],
        )
        check_wsi(wsi=wsi_file, magnification=configuration["magnification"])
        cell_segmentation.process_wsi(
            wsi_file,
            subdir_name=configuration["outdir_subdir"],
            geojson=configuration["geojson"],
            batch_size=configuration["batch_size"],
        )

    elif command.lower() == "process_dataset":
        cell_segmentation.logger.info("Processing whole dataset")
        if configuration["filelist"] is not None:
            if Path(configuration["filelist"]).suffix != ".csv":
                raise ValueError("Filelist must be a .csv file!")
            cell_segmentation.logger.info(
                f"Loading files from filelist {configuration['filelist']}"
            )
            wsi_filelist = load_wsi_files_from_csv(
                csv_path=configuration["filelist"],
                wsi_extension=configuration["wsi_extension"],
            )
            wsi_filelist = [
                Path(configuration["wsi_paths"]) / f
                if configuration["wsi_paths"] not in f
                else Path(f)
                for f in wsi_filelist
            ]
        else:
            cell_segmentation.logger.info(
                f"Loading all files from folder {configuration['wsi_paths']}. No filelist provided."
            )
            wsi_filelist = [
                f
                for f in sorted(
                    Path(configuration["wsi_paths"]).glob(
                        f"**/*.{configuration['wsi_extension']}"
                    )
                )
            ]
        for i, wsi_path in enumerate(wsi_filelist):
            wsi_path = Path(wsi_path)
            wsi_name = wsi_path.stem
            patched_slide_path = Path(configuration["patch_dataset_path"]) / wsi_name
            cell_segmentation.logger.info(f"File {i + 1}/{len(wsi_filelist)}: {wsi_name}")
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
                patch_size=1024,
                overlap=64
            )
