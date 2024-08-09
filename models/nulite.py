import copy
from collections import OrderedDict
from typing import List, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary

from models.encoders.fastvit import FastViTEncoder
from models.utils import Conv2DBlock
from nuclei_detection.utils.post_proc_nulite import DetectionCellPostProcessor


class NuLite(nn.Module):
    """NuFastViT

    Skip connections are shared between branches, but each network has a distinct encoder

    Args:
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        vit_structure (Literal["SAM-B", "SAM-L", "SAM-H"]): SAM model type
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        regression_loss (bool, optional): Use regressive loss for predicting vector components.
            Adds two additional channels to the binary decoder, but returns it as own entry in dict. Defaults to False.

    Raises:
        NotImplementedError: Unknown Fast-ViT backbone structure
    """

    def __init__(
            self,
            num_nuclei_classes: int,
            num_tissue_classes: int,
            vit_structure: Literal[
                "fastvit_t8", "fastvit_t12", "fastvit_s12", "fastvit_sa24", "fastvit_sa36", "fastvit_sa36"],
            drop_rate: int = 0,
    ):
        super().__init__()
        self.vit_structure = vit_structure
        self.embed_dims = []
        if vit_structure == "fastvit_t8":
            self._init_t8()
        elif vit_structure == "fastvit_t12":
            self._init_t12()
        elif vit_structure == "fastvit_s12":
            self._init_s12()
        elif vit_structure == "fastvit_sa12":
            self._init_sa12()
        elif vit_structure == "fastvit_sa24":
            self._init_sa24()
        elif vit_structure == "fastvit_sa36":
            self._init_sa36()
        elif vit_structure == "fastvit_ma36":
            self._init_ma36()
        else:
            raise NotImplementedError("Unknown Fast-ViT backbone structure")
        self.drop_rate = drop_rate
        self.num_nuclei_classes = num_nuclei_classes
        self.regression_loss = False
        self.encoder = FastViTEncoder(vit_structure)
        self.classifier_head = (
            nn.Linear(self.embed_dims[-1], num_tissue_classes)
            if num_tissue_classes > 0
            else nn.Identity()
        )
        # version with shared skip_connections
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, self.embed_dims[-4], 3, dropout=self.drop_rate),
        )  # skip connection after positional encoding, shape should be H, W, 64

        offset_branches = 0
        self.branches_output = {
            "nuclei_binary_map": 2 + offset_branches,
            "hv_map": 2,
            "nuclei_type_maps": self.num_nuclei_classes,
        }

        self.decoder = self.create_upsampling_branch()
        self.np_head = nn.Sequential(
            Conv2DBlock(2 * self.embed_dims[-4], self.embed_dims[-4], dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=self.embed_dims[-4],
                out_channels=2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        self.hv_head = nn.Sequential(
            Conv2DBlock(2 * self.embed_dims[-4], self.embed_dims[-4], dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=self.embed_dims[-4],
                out_channels=2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        self.tp_head = nn.Sequential(
            Conv2DBlock(2 * self.embed_dims[-4], self.embed_dims[-4], dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=self.embed_dims[-4],
                out_channels=self.num_nuclei_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )


    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False):
        """Forward pass

        Args:
            x (torch.Tensor): Images in BCHW style
            retrieve_tokens (bool, optional): If tokens of FastViT should be returned as well. Defaults to False.

        Returns:
            dict: Output for all branches:
                * tissue_types: Raw tissue type prediction. Shape: (B, num_tissue_classes)
                * nuclei_binary_map: Raw binary cell segmentation predictions. Shape: (B, 2, H, W)
                * hv_map: Binary HV Map predictions. Shape: (B, 2, H, W)
                * nuclei_type_map: Raw binary nuclei type preditcions. Shape: (B, num_nuclei_classes, H, W)
                * [Optional, if retrieve tokens]: tokens
        """

        out_dict = {}

        classifier_logits, _, z = self.encoder(x)
        out_dict["tissue_types"] = self.classifier_head(classifier_logits)

        z0, z1, z2, z3, z4 = x, *z

        decoder = self._forward_upsample(z1,z2,z3,z4, self.decoder)

        xt = self.decoder0(x)
        xt = torch.cat([xt, decoder], dim=1)
        out_dict["nuclei_binary_map"] = self.np_head(xt)

        out_dict["hv_map"] = self.hv_head(xt)
        out_dict["nuclei_type_map"] = self.tp_head(xt)

        if retrieve_tokens:
            out_dict["tokens"] = z4

        return out_dict

    def _forward_upsample(
            self,
            #z0: torch.Tensor,
            z1: torch.Tensor,
            z2: torch.Tensor,
            z3: torch.Tensor,
            z4: torch.Tensor,
            branch_decoder: nn.Sequential,
    ) -> torch.Tensor:
        """Forward upsample branch

        Args:
            z0 (torch.Tensor): Highest skip
            z1 (torch.Tensor): 1. Skip
            z2 (torch.Tensor): 2. Skip
            z3 (torch.Tensor): 3. Skip
            z4 (torch.Tensor): Bottleneck
            branch_decoder (nn.Sequential): Branch decoder network

        Returns:
            torch.Tensor: Branch Output
        """
        b5 = branch_decoder.bottleneck_upsampler(z4)
        b4 = branch_decoder.decoder4_upsampler(torch.cat([z3, b5], dim=1))
        b3 = branch_decoder.decoder3_upsampler(torch.cat([z2, b4], dim=1))
        b2 = branch_decoder.decoder2_upsampler(torch.cat([z1, b3], dim=1))
        b1 = branch_decoder.decoder1_upsampler(b2)
        return b1

    def create_upsampling_branch(self) -> nn.Module:
        """Create Upsampling branch

        Args:
            num_classes (int): Number of output classes

        Returns:
            nn.Module: Upsampling path
        """
        bottleneck_upsampler = nn.Sequential(
            Conv2DBlock(self.embed_dims[-1], self.embed_dims[-2], dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=self.embed_dims[-2],
                out_channels=self.embed_dims[-2],
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )

        decoder4_upsampler = nn.Sequential(
            Conv2DBlock(2 * self.embed_dims[-2], self.embed_dims[-2], dropout=self.drop_rate),
            Conv2DBlock(self.embed_dims[-2], self.embed_dims[-3], dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=self.embed_dims[-3],
                out_channels=self.embed_dims[-3],
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )

        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(2 * self.embed_dims[-3], self.embed_dims[-3], dropout=self.drop_rate),
            Conv2DBlock(self.embed_dims[-3], self.embed_dims[-4], dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=self.embed_dims[-4],
                out_channels=self.embed_dims[-4],
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )

        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(2 * self.embed_dims[-4], self.embed_dims[-4], dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=self.embed_dims[-4],
                out_channels=self.embed_dims[-4],
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )

        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(self.embed_dims[-4], self.embed_dims[-4], dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=self.embed_dims[-4],
                out_channels=self.embed_dims[-4],
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )



        decoder = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder4_upsampler", decoder4_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                ]
            )
        )

        return decoder

    def calculate_instance_map(
            self, predictions: OrderedDict, magnification: Literal[20, 40] = 40
    ) -> Tuple[torch.Tensor, List[dict]]:
        """Calculate Instance Map from network predictions (after Softmax output)

        Args:
            predictions (dict): Dictionary with the following required keys:
                * nuclei_binary_map: Binary Nucleus Predictions. Shape: (B, 2, H, W)
                * nuclei_type_map: Type prediction of nuclei. Shape: (B, self.num_nuclei_classes, H, W)
                * hv_map: Horizontal-Vertical nuclei mapping. Shape: (B, 2, H, W)
            magnification (Literal[20, 40], optional): Which magnification the data has. Defaults to 40.

        Returns:
            Tuple[torch.Tensor, List[dict]]:
                * torch.Tensor: Instance map. Each Instance has own integer. Shape: (B, H, W)
                * List of dictionaries. Each List entry is one image. Each dict contains another dict for each detected nucleus.
                    For each nucleus, the following information are returned: "bbox", "centroid", "contour", "type_prob", "type"
        """
        # reshape to B, H, W, C
        predictions_ = predictions.copy()
        predictions_["nuclei_type_map"] = predictions_["nuclei_type_map"].permute(
            0, 2, 3, 1
        )
        predictions_["nuclei_binary_map"] = predictions_["nuclei_binary_map"].permute(
            0, 2, 3, 1
        )
        predictions_["hv_map"] = predictions_["hv_map"].permute(0, 2, 3, 1)

        cell_post_processor = DetectionCellPostProcessor(
            nr_types=self.num_nuclei_classes, magnification=magnification, gt=False
        )
        instance_preds = []
        type_preds = []

        for i in range(predictions_["nuclei_binary_map"].shape[0]):
            pred_map = np.concatenate(
                [
                    torch.argmax(predictions_["nuclei_type_map"], dim=-1)[i]
                    .detach()
                    .cpu()[..., None],
                    torch.argmax(predictions_["nuclei_binary_map"], dim=-1)[i]
                    .detach()
                    .cpu()[..., None],
                    predictions_["hv_map"][i].detach().cpu(),
                ],
                axis=-1,
            )
            instance_pred = cell_post_processor.post_process_cell_segmentation(pred_map)
            instance_preds.append(instance_pred[0])
            type_preds.append(instance_pred[1])

        return torch.Tensor(np.stack(instance_preds)), type_preds

    def generate_instance_nuclei_map(
            self, instance_maps: torch.Tensor, type_preds: List[dict]
    ) -> torch.Tensor:
        """Convert instance map (binary) to nuclei type instance map

        Args:
            instance_maps (torch.Tensor): Binary instance map, each instance has own integer. Shape: (B, H, W)
            type_preds (List[dict]): List (len=B) of dictionary with instance type information (compare post_process_hovernet function for more details)

        Returns:
            torch.Tensor: Nuclei type instance map. Shape: (B, self.num_nuclei_classes, H, W)
        """
        batch_size, h, w = instance_maps.shape
        instance_type_nuclei_maps = torch.zeros(
            (batch_size, h, w, self.num_nuclei_classes)
        )
        for i in range(batch_size):
            instance_type_nuclei_map = torch.zeros((h, w, self.num_nuclei_classes))
            instance_map = instance_maps[i]
            type_pred = type_preds[i]
            for nuclei, spec in type_pred.items():
                nuclei_type = spec["type"]
                instance_type_nuclei_map[:, :, nuclei_type][
                    instance_map == nuclei
                    ] = nuclei

            instance_type_nuclei_maps[i, :, :, :] = instance_type_nuclei_map

        instance_type_nuclei_maps = instance_type_nuclei_maps.permute(0, 3, 1, 2)
        return torch.Tensor(instance_type_nuclei_maps)

    @staticmethod
    def reparameterize_model(m: torch.nn.Module) -> nn.Module:
        """Method returns a model where a multi-branched structure
            used in training is re-parameterized into a single branch
            for inference.

        Args:
            model: MobileOne model in train mode.

        Returns:
            MobileOne model in inference mode.
        """
        # Avoid editing original graph
        m = copy.deepcopy(m)
        for module in m.modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()
        return m

    def reparameterize_encoder(self):
        self.encoder.fast_vit = self.reparameterize_model(self.encoder.fast_vit)

    def _init_t8(self):
        self.embed_dims = [48, 96, 192, 384]

    def _init_t12(self):
        self.embed_dims = [64, 128, 256, 512]

    def _init_s12(self):
        self.embed_dims = [64, 128, 256, 512]

    def _init_sa12(self):
        self.embed_dims = [64, 128, 256, 512]

    def _init_sa24(self):
        self.embed_dims = [64, 128, 256, 512]

    def _init_sa36(self):
        self.embed_dims = [64, 128, 256, 512]

    def _init_ma36(self):
        self.embed_dims = [76, 152, 304, 608]


if __name__ == '__main__':

    x = torch.randn((1, 3, 256, 256)).cuda()
    model = NuLite(6,19,"fastvit_ma36").cuda()
    summary(model, (1,3,256,256),device="cuda")
    #print(model(x))