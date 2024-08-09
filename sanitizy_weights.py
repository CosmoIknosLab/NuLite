import torch

base_config = {
    'data.dataset': 'PanNuke',
    'data.num_nuclei_classes': 6,
    'data.num_tissue_classes': 19,
    'training.drop_rate': 0,
    'transformations.normalize.mean': [
        0.5,
        0.5,
        0.5
    ],
    'dataset_config.tissue_types.Adrenal_gland': 0,
    'dataset_config.tissue_types.Bile-duct': 1,
    'dataset_config.tissue_types.Bladder': 2,
    'dataset_config.tissue_types.Breast': 3,
    'dataset_config.tissue_types.Cervix': 4,
    'dataset_config.tissue_types.Colon': 5,
    'dataset_config.tissue_types.Esophagus': 6,
    'dataset_config.tissue_types.HeadNeck': 7,
    'dataset_config.tissue_types.Kidney': 8,
    'dataset_config.tissue_types.Liver': 9,
    'dataset_config.tissue_types.Lung': 10,
    'dataset_config.tissue_types.Ovarian': 11,
    'dataset_config.tissue_types.Pancreatic': 12,
    'dataset_config.tissue_types.Prostate': 13,
    'dataset_config.tissue_types.Skin': 14,
    'dataset_config.tissue_types.Stomach': 15,
    'dataset_config.tissue_types.Testis': 16,
    'dataset_config.tissue_types.Thyroid': 17,
    'dataset_config.tissue_types.Uterus': 18,
    'dataset_config.nuclei_types.Background': 0,
    'dataset_config.nuclei_types.Neoplastic': 1,
    'dataset_config.nuclei_types.Inflammatory': 2,
    'dataset_config.nuclei_types.Connective': 3,
    'dataset_config.nuclei_types.Dead': 4,
    'dataset_config.nuclei_types.Epithelial': 5
}

model = 'NuLite-T'
backbone = "fastvit_s12"
weights_path = f'/work/cristian/weights/NNuLite/{backbone}/latest_checkpoint.pth'
save_path = f'/work/cristian/weights/NNuLite/{backbone}/{model}-Weights.pth'

old_weights = torch.load(weights_path)

new_weights = dict()
new_weights["model_state_dict"] = old_weights["model_state_dict"]
new_weights["arch"] = model
base_config["model.backbone"] = backbone
new_weights["config"] = base_config
torch.save(new_weights, save_path)
