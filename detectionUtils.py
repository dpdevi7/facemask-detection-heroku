import torch, torchvision
import cv2
import numpy as np
from typing import List, Optional
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

# get Device
def getDevice():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# pretrained inbuilt pytorch ssd-mobilenetV3 model
def loadModelPretrained():
    SSD_MODEL = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=False)
    checkPoint = torch.load(
        "model-dir/facemaskDetectionSSD_320x320.pth",
        map_location=getDevice())

    SSD_MODEL.load_state_dict(checkPoint['model_state_dict'])
    SSD_MODEL.eval()
    SSD_MODEL.to(getDevice())

    return SSD_MODEL




def _xavier_normal_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)
def _kaiming_normal_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)


# MobileNetV3 Custom build for facemask classification
class CustomMV3(nn.Module):
    def __init__(self, base):
        super().__init__()

        self.features = base.features
        self.avgPool = base.avgpool
        self.flatten = nn.Flatten()
        self.classfier = nn.Sequential(

            nn.Linear(960, 1280, bias=True),
            nn.Hardswish(inplace=False),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=3, bias=True)

        )

    def forward(self, x):
        out = self.features(x)
        out = self.avgPool(out)
        out = self.flatten(out)
        out = self.classfier(out)
        return out

class SSDFeatureExtractorMobilenetV3(torch.nn.Module):

    def __init__(self, base):
        super(SSDFeatureExtractorMobilenetV3, self).__init__()

        self.featuresOneFromBase = nn.Sequential(
            *base[:5]
        )

        self.featuresTwoFromBase = nn.Sequential(
            *base[5:10]  # until InvertedResidual 9
        )

        self.featuresThreeFromBase = nn.Sequential(
            *base[10:]  # until InvertedResidual 10 --> 16
        )

        fc1 = nn.Sequential(
            nn.Conv2d(in_channels=960, out_channels=64, kernel_size=1, padding=0, stride=1),  # FC6 with atrous
            nn.BatchNorm2d(num_features=64),
            nn.Hardswish(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0, stride=1),  # FC6 with atrous
            nn.BatchNorm2d(num_features=128),
            nn.Hardswish(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding=0, stride=1),  # FC6 with atrous
            nn.BatchNorm2d(num_features=256),

        )

        fc2 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1, padding=0, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.Hardswish(inplace=True),

            nn.Conv2d(32, 64, kernel_size=1, padding=0, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.Hardswish(inplace=True),

            nn.Conv2d(64, 128, kernel_size=1, padding=0, stride=2),
            nn.BatchNorm2d(num_features=128),
        )
        fc3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, padding=0, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.Hardswish(inplace=True),
        )
        fc4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(num_features=32),
            nn.Hardswish(inplace=True),
        )
        fc5 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, padding=0, stride=2),
            nn.BatchNorm2d(num_features=16),
            nn.Hardswish(inplace=True),
        )

        _xavier_normal_init(fc1)
        _xavier_normal_init(fc2)
        _xavier_normal_init(fc3)
        _xavier_normal_init(fc4)
        _xavier_normal_init(fc5)

        self.extra = nn.ModuleList([fc1, fc2, fc3, fc4, fc5])

        self.scale_weight = nn.Parameter(torch.ones(960) * 200)

    def forward(self, x):
        output = []

        outOne = self.featuresOneFromBase(x)
        output.append(outOne)

        out = self.featuresTwoFromBase(outOne)
        output.append(out)

        out = self.featuresThreeFromBase(out)
        output.append(out)

        for i, block in enumerate(self.extra):
            out = block(out)
            output.append(out)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])

def loadMobileNetV3(path, model):
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.train()
    return model

def get_ssd_backbone():
    pretrainedMobilenetV3 = torchvision.models.mobilenet_v3_large(pretrained=False)
    pretrainedMobilenetV3 = CustomMV3(pretrainedMobilenetV3)
    pretrainedMobilenetV3 = pretrainedMobilenetV3.features

    for i, b in enumerate(pretrainedMobilenetV3):
        for param in b.parameters():
            param.requires_grad = False

    # TODO: freeze initial layers of MobileNetV3
    return SSDFeatureExtractorMobilenetV3(pretrainedMobilenetV3)

def get_ssd_model(num_classes, size):
    backbone = get_ssd_backbone()
    # total no. of features = 5
    # for highres features = 6, change aspect ration numbers accordingly
    anchor_generator = torchvision.models.detection.anchor_utils.DefaultBoxGenerator([[2, 3],
                                                                                      [2, 3],
                                                                                      [2, 3],
                                                                                      [2, 3],
                                                                                      [2, 3],
                                                                                      [2, 3],
                                                                                      [2, 3], [2]],
                                                                                     scales=[0.07, 0.15, 0.33, 0.51,
                                                                                             0.69, 0.87, 1.05, 1.3,
                                                                                             1.5])

    defaults = {
        # Rescale the input in a way compatible to the backbone

        "image_mean": [0.48235, 0.45882, 0.40784],
        "image_std": [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
        # undo the 0-1 scaling of toTensor
    }

    kwargs = {**defaults}
    SSD_CUSTOM = torchvision.models.detection.ssd.SSD(backbone, anchor_generator, (size, size), num_classes, **kwargs)

    return SSD_CUSTOM

def get_ssd_from_checkpoint(modelFile, SSD_MODEL):
    checkpoint = torch.load(modelFile, map_location=torch.device('cpu'))
    SSD_MODEL.load_state_dict(checkpoint['model_state_dict'])
    SSD_MODEL.eval()
    return SSD_MODEL

def get_ssd_from_checkpoint_scripted_module(modelFile, SSD_MODEL):
    checkpoint = torch.load(modelFile, map_location=torch.device('cpu'))
    SSD_MODEL.load_state_dict(checkpoint['model_state_dict'])
    SSD_MODEL.eval()
    SCRIPTED_SSD_MODEL = torch.jit.script(SSD_MODEL)
    return SCRIPTED_SSD_MODEL

def detectMaskNonScriptedModule(model, imgTensor):
    out = model(imgTensor)
    keepIndex = torchvision.ops.nms(out[0]['boxes'], out[0]['scores'], iou_threshold=0.8)
    bboxTensors = out[0]['boxes'][keepIndex]
    labelsTensors = out[0]['labels'][keepIndex]
    scoresTensors = out[0]['scores'][keepIndex]

    boxes = bboxTensors.detach().cpu().numpy().astype(np.int)
    labels = labelsTensors.detach().cpu().numpy()
    scores = scoresTensors.detach().cpu().numpy()

    return boxes.tolist(), labels.tolist(), scores.tolist()

def detectMaskScriptedModule(model, imgTensor):
    # 1. scripted module, take list of tensors
    out = model([imgTensor])
    # 2. prediction = (loss, detections)
    predLosses = out[0]
    predDetections = out[1][0]
    # 3. NMS
    keepIndex = torchvision.ops.nms(predDetections['boxes'], predDetections['scores'], iou_threshold=0.2)
    # 4. get Detections after NMS
    bboxTensors = predDetections['boxes'][keepIndex]
    labelsTensors = predDetections['labels'][keepIndex]
    scoresTensors = predDetections['scores'][keepIndex]
    # 5. serve to response
    boxes = bboxTensors.detach().cpu().numpy().astype(np.int)
    labels = labelsTensors.detach().cpu().numpy()
    scores = scoresTensors.detach().cpu().numpy()

    return boxes.tolist(), labels.tolist(), scores.tolist()

def getAllowedFileExtension(filename):
    extension = filename.split('.')[-1]
    if extension in ALLOWED_EXTENSIONS:
        return True
    else:
        return False

def readImageNonScriptedModule(imageBytes):
    imageBytesNumpy = np.fromstring(imageBytes, np.uint8)
    img = cv2.imdecode(imageBytesNumpy, cv2.IMREAD_COLOR).astype(np.float32)
    img = cv2.resize(img, (320, 320))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB).astype(np.float32)
    img = img / 255.0
    imgTensor = torch.tensor(img, dtype=torch.float32)
    imgTensor = imgTensor.permute(2,0,1).unsqueeze(dim=0)

    return imgTensor

def readImageScriptedModule(imageBytes):
    imageBytesNumpy = np.fromstring(imageBytes, np.uint8)
    img = cv2.imdecode(imageBytesNumpy, cv2.IMREAD_COLOR).astype(np.float32)
    img = img / 255.0
    imgTensor = torch.tensor(img, dtype=torch.float32)
    imgTensor = imgTensor.permute(2,0,1)

    return imgTensor





if __name__ == '__main__':
    pass
