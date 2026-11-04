import torch
from models.lcmam_variants import AquaMamNet

def build_model(num_classes=1, input_size=128, pretrained=False, in_channels=3, **kwargs):
    return AquaMamNet(in_channels=in_channels, num_classes=num_classes, base_c=32, input_size=input_size)
