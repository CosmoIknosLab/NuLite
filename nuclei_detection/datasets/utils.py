dict_tissue = {
        "Adrenal_gland": 0,
        "Bile-duct": 1,
        "Bladder": 2,
        "Breast": 3,
        "Cervix": 4,
        "Colon": 5,
        "Esophagus": 6,
        "HeadNeck": 7,
        "Kidney": 8,
        "Liver": 9,
        "Lung": 10,
        "Ovarian": 11,
        "Pancreatic": 12,
        "Prostate": 13,
        "Skin": 14,
        "Stomach": 15,
        "Testis": 16,
        "Thyroid": 17,
        "Uterus": 18
    }

reversed_dict_tissue = {value: key for key, value in dict_tissue.items()}


def encoder_tissue(tissue_name):
    return dict_tissue[tissue_name]

def decoder_tissue(encoded_value):
    return reversed_dict_tissue.get(encoded_value, "Unknown Tissue Code")