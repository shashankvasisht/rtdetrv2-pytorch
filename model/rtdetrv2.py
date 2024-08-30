import torch.nn as nn

# from .backbone import PResNet
# from .encoder import HybridEncoder
# from .decoder import RTDETRTransformerv2


class RTDETR(nn.Module):

    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self


# class RTDETR(nn.Module):

#     def __init__(self, data, **kwargs):
#         super().__init__()
#         self.backbone = PResNet(depth=50, return_idx=[1, 2, 3])
#         self.encoder = HybridEncoder()
#         self.decoder = RTDETRTransformerv2(
#             num_classes=data.c,
#             feat_channels=[256, 256, 256],
#             num_points=[4, 4, 4],
#         )

#     def forward(self, x, targets=None):
#         x = self.backbone(x)
#         x = self.encoder(x)
#         x = self.decoder(x, targets)

#         return x
