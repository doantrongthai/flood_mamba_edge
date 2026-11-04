import torch
from models.lcmam_variants import LCMam_Step1

def build_model(num_classes=1, input_size=128, pretrained=False, in_channels=3, **kwargs):
    return LCMam_Step1(in_channels=in_channels, num_classes=num_classes, base_c=32, input_size=input_size)
