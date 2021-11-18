"""
Created on Wed Dec 16 2020
@author: xxx
"""
from models import *
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from IPython.display import HTML
import torchvision.models as models
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from OOD_Metric import evaluate_detection

def init_setting(seed, alg, dataset, model, name, detector, OOD_list):

	# init seed
	if seed != 0:
		torch.manual_seed(seed)

	# Address
	if not os.path.isdir('results'):
		os.mkdir('results')
	if not os.path.isdir('results/' + alg):
		os.mkdir('results/' + alg)
	if not os.path.isdir('checkpoint'):
		os.mkdir('checkpoint')

	filename = (alg + '_' + dataset + '_' + model + '_'  +  name)
	pth_name = ('checkpoint/' + filename + '.pth')
	log_name = ('results/' + alg + '/' + filename + '_LOG.csv')
	detection_name = ('results/' + alg + '/' + filename + '_'  +  detector + '_LOG.csv')
	table_name = ('results/' + alg + '/' + dataset + '.csv')
		
	if not os.path.exists(table_name):
		with open(table_name, 'w') as logfile:
			logwriter = csv.writer(logfile, delimiter=',')
			info = ['algorithm', 'detector', 'model', 'dataset', 'name','ACC', 'ECE', 'ys']
			info.extend(OOD_list)
			logwriter.writerow(info)

	if not os.path.exists(log_name):
		with open(log_name, 'w') as logfile:
			logwriter = csv.writer(logfile, delimiter=',')
			logwriter.writerow(['epoch', 'lr', 'train loss', 'train acc', 'test loss', 'test acc'])

	if not os.path.exists(detection_name):
		with open(detection_name, 'w') as logfile:
			logwriter = csv.writer(logfile, delimiter=',')
			logwriter.writerow(['OOD', 'detector', 'AUROC', 'AUPR(IN)', 'AUPR(OUT)', 'FPR(95)', 'Detection'])

	return log_name, detection_name, table_name, pth_name

def build_model(model, dataset, num_classes=10):

	if dataset == 'CIFAR10' or dataset == 'SVHN':
		if model == 'resnet':
			net = ResNet18(num_classes=num_classes)
		elif model == 'densenet':
			net = DenseNet121(num_classes=num_classes)
		elif model == 'vgg':
			net = VGG('VGG19',num_classes=num_classes)
		elif model == 'shufflenet':
			net = MobileNetV2(num_classes=num_classes)
		elif model == 'mobilenet':
			net = ShuffleNetV2(num_classes=num_classes)
		elif model == 'senet':
			net = SENet18(num_classes=num_classes)

	net.cuda()
	net = torch.nn.DataParallel(net)
	cudnn.benchmark = True
	return net

def load_pretrained(net, pth_name):
	net.load_state_dict(torch.load(pth_name))

def save_model(pth_name, net, save):
	if save:
		torch.save(net.state_dict(), pth_name)

def save_result(log_name, epoch, train_loss, train_acc, test_loss, test_acc, optimizer):
	with open(log_name, 'a') as logfile:
		logwriter = csv.writer(logfile, delimiter=',')
		logwriter.writerow([epoch, optimizer.state_dict()['param_groups'][0]['lr'],
			train_loss, train_acc, test_loss, test_acc])
		
def adjust_learning_rate(decay_epochs, optimizer, epoch):
	if epoch in decay_epochs:
		for param_group in optimizer.param_groups:
			new_lr = param_group['lr'] * 0.1
			param_group['lr'] = new_lr

def imshow(img, figpath='NULL', save=False, r='none'):

	if r != 'none':
		h = 32
		mean, std = data_loader.get_known_mean_std(r)
		t_mean = torch.FloatTensor(mean).view(3,1,1).expand(3,h,h)
		t_std = torch.FloatTensor(std).view(3,1,1).expand(3,h,h)
		img = img * t_std + t_mean

	#img = img / 2 + 0.5
	#img = (img + 1)/2
	
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)))

	fig = plt.gcf()
	s = 96
	fig.set_size_inches(960/s, 960/s)
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())
	plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
	plt.margins(0,0)
	if save:
		plt.savefig(figpath, dpi=s)
	else:
		plt.show()

def get_mu(testloader, net, num_classes):
	net.eval()
	total = 0
	mu = torch.zeros(num_classes).cuda()
	with torch.no_grad():
		for idx, (inputs_ID, targets) in enumerate(testloader):
			inputs_ID, targets = inputs_ID.cuda(), targets.cuda()
			outputs_ID = net(inputs_ID)
			mu += torch.mean(outputs_ID, 0)
			total += targets.size(0)
	mu = mu / total
	return mu

def get_std(testloader, net, mu, num_classes):
	net.eval()
	total = 0
	std = torch.zeros(num_classes, num_classes).cuda()
	with torch.no_grad():
		for idx, (inputs_ID, targets) in enumerate(testloader):
			inputs_ID, targets = inputs_ID.cuda(), targets.cuda()
			outputs_ID = net(inputs_ID)
			outputs = outputs_ID.data - mu
			for i in range(0, outputs.size(0)):
				std += torch.matmul(outputs[i].unsqueeze(1), outputs[i].unsqueeze(0))
			total += targets.size(0)
	std = std / total
	return std
