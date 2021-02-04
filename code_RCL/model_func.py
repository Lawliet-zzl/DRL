"""
Created on Wed Dec 16 2020
@author: Zhilin Zhao
"""
from models import *
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from IPython.display import HTML
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

def build_model(model, num_classes=10):
	if model == 'ResNet18':
		net = ResNet18(num_classes=num_classes)
	elif model == 'ResNet34':
		net = ResNet34(num_classes=num_classes)
	elif model == 'ResNet50':
		net = ResNet50(num_classes=num_classes)
	elif model == 'DenseNet121':
		net = DenseNet121(num_classes=num_classes)
	elif model == 'PreActResNet':
		net = PreActResNet18(num_classes=num_classes)
	elif model == 'VGG19':
		net = VGG('VGG19',num_classes=num_classes)
	elif model == 'ResNeXt29_2x64d':
		net = ResNeXt29_2x64d(num_classes=num_classes)
	elif model == 'DPN92':
		net = DPN92(num_classes=num_classes)
	elif model == 'RegNetX_200MF':
		net = RegNetX_200MF(num_classes=num_classes)
	elif model == 'WideResNet':
		net = Wide_ResNet(num_classes=num_classes)
	elif model == 'MobileNetV2':
		net = MobileNetV2(num_classes=num_classes)
	elif model == 'Inception':
		net = GoogLeNet(num_classes=num_classes)
	elif model == 'ShuffleNetV2':
		net = ShuffleNetV2(num_classes=num_classes)
	elif model == 'SENet':
		net = SENet18(num_classes=num_classes)
	elif model == 'DPN26':
		net = DPN26(num_classes=num_classes)
	elif model == 'EfficientNet':
		net = EfficientNetB0(num_classes=num_classes)
	elif model == 'LeNet':
		net = LeNet(num_classes=num_classes, channels=1)

	net.cuda()
	net = torch.nn.DataParallel(net)
	cudnn.benchmark = True
	return net