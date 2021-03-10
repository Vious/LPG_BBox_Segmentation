import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """
    def __init__(self, inChannels, outChannels, rates):
        super(ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(inChannels, outChannels, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        return sum([stage(features) for stage in self.children()])
    

class DeepLabV2_Resnet101(nn.Sequential):
    def __init__(self, layers=101, num_classes=21, atrous_rates=[6, 12, 18, 24], pretrained=True):
        super(DeepLabV2_Resnet101, self).__init__()
        assert layers in [50, 101, 152]
        if layers == 50:
            resnet = models.resnet50()
            if pretrained:
                state_dict = torch.load('./initialModel/resnet50-19c8e357.pth', map_location=torch.device('cpu'))
                resnet.load_state_dict(state_dict, strict=False)
        elif layers == 101:
            resnet = models.resnet101()
            if pretrained:
                state_dict = torch.load('./initialModel/resnet101-5d3b4d8f.pth', map_location=torch.device('cpu'))
                resnet.load_state_dict(state_dict, strict=False)
                print('Using ImageNet pretrained initial!!!')
        else:
            resnet = models.resnet152()
            if pretrained:
                state_dict = torch.load('./initialModel/resnet152-b121ed2d.pth', map_location=torch.device('cpu'))
                resnet.load_state_dict(state_dict, strict=False)

        layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
            # resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool) 
        layer1, layer2, layer3, layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        featureDimensions = 2048
        self.add_module('layer0', layer0)
        self.add_module('layer1', layer1)
        self.add_module('layer2', layer2)
        self.add_module('layer3', layer3)
        self.add_module('layer4', layer4)
        self.add_module('aspp', ASPP(featureDimensions, num_classes, atrous_rates))


        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


if __name__ == "__main__":
    model = DeepLabV2_Resnet101(layers=101, classes=21, pretrained=False).cuda()
    print(model)
    image = torch.randn(1, 3, 513, 513).cuda()


    # print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)