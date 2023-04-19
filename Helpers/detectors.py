"""
From https://github.com/coff-tea/whistle_detector
""" 


import math
import torch
import torch.nn as nn
import torchvision.models as models


#===============================================================================================
# Simple convolution network used as a detection, with specifications given about number of layers as input
class SimpleDetector(nn.Module):
    def __init__(self, chs_in, dim_in, dropout, filter_num, filter_size, nodes, gap, classes=1):
        super(SimpleDetector, self).__init__()
        conv_layers = []
        chs_curr = chs_in
        dim_curr = dim_in
        for i in range(len(filter_num)):
            conv_layers.append(nn.Conv2d(chs_curr, filter_num[i], filter_size[i]))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(2))
            chs_curr = filter_num[i]
            dim_curr = math.floor((dim_curr-2)/2)
        self.conv = nn.Sequential(*conv_layers)
        self.flat = nn.Flatten()
        if gap:
            self.dense = nn.AdaptiveAvgPool1d(classes)
        else:
            dense_layers = []
            dim_curr = dim_curr*dim_curr*chs_curr
            for i in range(len(nodes)):
                dense_layers.append(nn.Linear(dim_curr, nodes[i]))
                dense_layers.append(nn.ReLU())
                dense_layers.append(nn.Dropout(dropout))
                dim_curr = nodes[i]
            dense_layers.append(nn.Linear(dim_curr, classes))
            self.dense = nn.Sequential(*dense_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.flat(x)
        x = self.dense(x)
        return x



#===============================================================================================
#### FUNCTION: make_detector ####
# Create and return an image-classification oriented model.
# PARAMETERS
# ... Required
#   - model_name (dictates the type of model)
#   - chs_in (channels in the input image)
#   - dim_in (DxD shape of input image)
# ... Default given
#   - freeze (freeze model parameters up to a certain point, model-dependent, only for some models)
#            [default freezes nothing]
#   - gap (replace final layer with global average pooling, only for some models)
#         [default FALSE]
#   - dropout (dropout probability where necessary)
#             [default 0.5]
#   - rgb_sel (choose one of the RGB channel's pre-trained parameters to replicate)
#             [default R (red)]
#   - classes (number of classes to output)
#             [default 1 (detector rather than classifier)]
#   - pre (used pretrained parameters for pretrained models rather than random)
#         [default TRUE]
def make_detector(model_name, chs_in, dim_in, freeze=0, gap=False, dropout=0.5, rgb_sel=0, classes=1, pre=True):
    model = None
    ####### VGG16 REPRODUCE FROM https://arxiv.org/abs/2211.15406
    if model_name == "vgg16tf":
        if pre:
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        else:
            model = models.vgg16()
        model.classifier = nn.Sequential(
            nn.Linear(25088, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, classes))
    elif model_name == "vgg16tfd":
        if pre:
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        else:
            model = models.vgg16()
        model.classifier = nn.Sequential(
            nn.Linear(25088, 50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(20, classes))
    ####### SIMPLE MODEL
    elif model_name == "simple":
        filter_num = [16, 32, 64, 64, 64]
        filter_size = [3, 3, 3, 3, 3]
        nodes = [256, 64, 16]
        model = SimpleDetector(chs_in, dim_in, dropout, filter_num, filter_size, nodes, gap, classes=classes)
    ####### VGG MODELS (16, 16_bn, 19, 19_bn)
    elif "vgg" in model_name:
        if model_name == "vgg16":
            if pre:
                model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            else:
                model = models.vgg16()
            fr_layers = [0, 5, 10, 17, 24, -1]
        elif model_name == "vgg16bn":
            if pre:
                model = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
            else:
                model = models.vgg16_bn()
            fr_layers = [0, 7, 14, 24, 34, -1]
        elif model_name == "vgg19":
            if pre:
                model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
            else:
                model = models.vgg19()
            fr_layers = [0, 5, 10, 19, 28, -1]
        else:
            if pre:
                model = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)
            else:
                model = models.vgg19_bn()
            fr_layers = [0, 7, 14, 27, 40, -1]
        hold = model.features[0].weight.data[:, rgb_sel:rgb_sel+1, :, :]
        model.features[0] = nn.Conv2d(chs_in, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        for ch in range(chs_in):
            model.features[0].weight.data[:, ch:ch+1, :, :] = hold
        if freeze > 0:
            for i in range(len(fr_layers)):
                if i == len(fr_layers)-1:
                    if i < freeze:
                        for param in model.classifier[0:6].parameters():
                            param.requires_grad = False
                elif i < freeze:
                    start = fr_layers[i]
                    end = fr_layers[i+1]
                    if end == -1:
                        for param in model.features[start:].parameters():
                            param.requires_grad = False
                    else:
                        for param in model.features[start:end].parameters():
                            param.requires_grad = False
        if gap:
            model.avgpool = nn.Flatten()    # Now a misnomer
            model.classifier = nn.AdaptiveAvgPool1d(classes)
        else:
            model.classifier[6] = nn.Linear(4096, classes)
    ####### RESNET MODELS (50, 101, 152)
    elif "res" in model_name:
        if model_name == "res50":
            if pre:
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                model = models.resnet50()
        elif model_name == "res101":
            if pre:
                model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            else:
                model = models.resnet101()
        else:
            if pre:
                model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
            else:
                model = models.resnet152()
        hold = model.conv1.weight.data[:, rgb_sel:rgb_sel+1, :, :]
        model.conv1 = nn.Conv2d(chs_in, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        for ch in range(chs_in):
            model.conv1.weight.data[:, ch:ch+1, :, :] = hold
        if freeze > 0:
            for param in model.conv1.parameters():
                param.requires_grad = False
            for param in model.bn1.parameters():
                param.requires_grad = False
        if freeze > 1:
            for param in model.layer1.parameters():
                param.requires_grad = False
        if freeze > 2:
            for param in model.layer2.parameters():
                param.requires_grad = False
        if freeze > 3:
            for param in model.layer3.parameters():
                param.requires_grad = False
        if freeze > 4:
            for param in model.layer4.parameters():
                param.requires_grad = False
        if gap:
            model.avgpool = nn.Flatten()
            model.fc = nn.AdaptiveAvgPool1d(classes)
        else:
            model.fc = nn.Linear(2048, classes, bias=True)
    ####### DENSENET MODELS (161, 169, 201)
    elif "dense" in model_name:
        if model_name == "dense161":
            if pre:
                model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
            else:
                model = models.densenet161()
            filter_num = 96
            hidden = 2208
        elif model_name == "dense169":
            if pre:
                model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
            else:
                model = models.densenet169()
            filter_num = 64
            hidden = 1664
        else:
            if pre:
                model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
            else:
                model = models.densenet201()
            filter_num = 64
            hidden = 1920
        hold = model.features.conv0.weight.data[:, rgb_sel:rgb_sel+1, :, :]
        model.features.conv0 = nn.Conv2d(chs_in, filter_num, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        for ch in range(chs_in):
            model.features.conv0.weight.data[:, ch:ch+1, :, :] = hold
        if freeze > 0:
            for param in model.features.conv0.parameters():
                param.requires_grad = False
            for param in model.features.norm0.parameters():
                param.requires_grad = False
        if freeze > 1:
            for param in model.features.denseblock1.parameters():
                param.requires_grad = False
        if freeze > 2:
            for param in model.features.transition1.parameters():
                param.requires_grad = False
        if freeze > 3:
            for param in model.features.denseblock2.parameters():
                param.requires_grad = False
        if freeze > 4:
            for param in model.features.transition2.parameters():
                param.requires_grad = False
        if freeze > 5:
            for param in model.features.denseblock3.parameters():
                param.requires_grad = False
        if freeze > 6:
            for param in model.features.transition3.parameters():
                param.requires_grad = False
        if freeze > 7:
            for param in model.features.denseblock4.parameters():
                param.requires_grad = False
        if freeze > 8:
            for param in model.features.norm5.parameters():
                param.requires_grad = False
        if gap:
            model.classifier = nn.AdaptiveAvgPool1d(classes)
        else:
            model.classifier = nn.Linear(hidden, classes, bias=True)
    return model
