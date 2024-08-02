"""
In this version I used timm.
Probably, I have to add original code of FastViT
"""
import torch.nn as nn
import timm


class FastViTEncoder(nn.Module):

    def __init__(self, vit_structure, pretrained=True):
        super(FastViTEncoder, self).__init__()

        self.fast_vit = timm.create_model(
            f'{vit_structure}.apple_in1k',
            features_only=True,
            pretrained=pretrained
        )

        self.avg_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    def forward(self, x):
        extracted_layers = self.fast_vit(x)
        return self.avg_pooling(extracted_layers[-1]), extracted_layers[-1], extracted_layers
