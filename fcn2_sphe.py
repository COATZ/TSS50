from typing import Optional

from torch import nn

import resnet_sphe as resnet_sphe
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation._utils import _SimpleSegmentationModel, _load_weights

from DeformConv2d_sphe import DeformConv2d_sphe

from operator import eq

__all__ = ["FCN", "fcn_resnet50", "fcn_resnet101"]


model_urls = {
    "fcn_resnet50_coco": "https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth",
    "fcn_resnet101_coco": "https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth",
}


class FCN(_SimpleSegmentationModel):
    """
    Implements FCN model from
    `"Fully Convolutional Networks for Semantic Segmentation"
    <https://arxiv.org/abs/1411.4038>`_.
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    pass


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]
        super().__init__(*layers)


class FCNHead_sphe(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            DeformConv2d_sphe(in_channels, inter_channels, 3, padding=1, bias=False),
            # nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]
        super().__init__(*layers)


def _fcn_resnet(
    backbone: resnet_sphe.ResNet,
    num_classes: int,
    aux: Optional[bool],
) -> FCN:
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux else None
    classifier = FCNHead(2048, num_classes)
    return FCN(backbone, classifier, aux_classifier)


def _fcn_resnet_sphe(
    backbone: resnet_sphe.ResNet,
    num_classes: int,
    aux: Optional[bool],
    iFCNHead_sphe: bool = False,
) -> FCN:
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    if iFCNHead_sphe:
        aux_classifier = FCNHead_sphe(1024, num_classes) if aux else None
        classifier = FCNHead_sphe(2048, num_classes)
    else:
        aux_classifier = FCNHead(1024, num_classes) if aux else None
        classifier = FCNHead(2048, num_classes)
    return FCN(backbone, classifier, aux_classifier)


def fcn_resnet18(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    pretrained_backbone: bool = True,
) -> FCN:
    """Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """
    if pretrained:
        aux_loss = True
        pretrained_backbone = False

    backbone = resnet_sphe.resnet18(pretrained=pretrained_backbone)
    model = _fcn_resnet(backbone, num_classes, aux_loss)

    # if pretrained:
    #     arch = "fcn_resnet18_coco"
    #     _load_weights(arch, model, model_urls.get(arch, None), progress)
    return model


def fcn_resnet50(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    pretrained_backbone: bool = True,
    nodilation: bool = False,
    iFCNHead_sphe: bool = False,
    iFL_sphe: bool = False,
    iSYMBOL=eq,
    LAYER_CCOUNT: int = 0,
    net_opt="",
) -> FCN:
    """Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """
    if pretrained:
        aux_loss = True
        pretrained_backbone = False

    if nodilation:
        backbone = resnet_sphe.resnet50(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, False, False], iFL_sphe=iFL_sphe, iSYMBOL=iSYMBOL, LAYER_CCOUNT=LAYER_CCOUNT)
    else:
        backbone = resnet_sphe.resnet50(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True], iFL_sphe=iFL_sphe, iSYMBOL=iSYMBOL, LAYER_CCOUNT=LAYER_CCOUNT)

    model = _fcn_resnet_sphe(backbone, num_classes, aux_loss, iFCNHead_sphe)

    if pretrained:
        arch = "fcn_resnet50_coco"
        _load_weights(arch, model, model_urls.get(arch, None), progress)
    return model


def fcn_resnet101(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    pretrained_backbone: bool = True,
) -> FCN:
    """Constructs a Fully-Convolutional Network model with a ResNet-101 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """
    if pretrained:
        aux_loss = True
        pretrained_backbone = False

    backbone = resnet_sphe.resnet101(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True])
    model = _fcn_resnet(backbone, num_classes, aux_loss)

    if pretrained:
        arch = "fcn_resnet101_coco"
        _load_weights(arch, model, model_urls.get(arch, None), progress)
    return model
