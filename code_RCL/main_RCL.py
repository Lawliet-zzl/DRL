"""
Created on Wed Dec 16 2020
@author: Zhilin Zhao
"""
from __future__ import print_function

import argparse
import csv
import os
import math
import time

import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import utils as vutils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from IPython.display import HTML
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from tqdm import tqdm

import data_loader
import OOD_Metric
from model_func import build_model

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--seed', default=20201120, type=int, help='random seed')
parser.add_argument('--pretrained-model', default="ResNet18", type=str, help='model type (default: ResNet18)')
parser.add_argument('--model', default="ResNet18", type=str, help='model type (default: ResNet18)')
parser.add_argument('--dataset', default="CIFAR10", type=str, help='dataset')
parser.add_argument('--num-classes', default=10, type=int, help='the number of classes (default: 10)')
parser.add_argument('--prename', default='0', type=str, help='name of run (default: 10)')
parser.add_argument('--name', default='0', type=str, help='name of run (default: 10)')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate (default: 0.1)')
parser.add_argument("--decay_epochs", nargs="+", type=int, default=[100, 150], help="decay learning rate by decay_rate at these epochs")
parser.add_argument('--decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='total epochs to run')

parser.add_argument('--alpha', default=0.1, type=float, help='alpha (default: 0.1)')
parser.add_argument('--eps', default=0.001, type=float, help='eps (default: 0.001)')
parser.add_argument('--precision', default=100000, type=float)


args = parser.parse_args()
args.num_classes = 100 if args.dataset == 'CIFAR100' else 10
best_acc = 0.0

class CRLoss(nn.Module):
	"""docstring for CRLoss"""
	def __init__(self, alpha = 0.1, eps = 0.1, mu = 0.1, std=0.1):
		super(CRLoss, self).__init__()
		self.CELoss = nn.CrossEntropyLoss()
		self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
		self.alpha = alpha
		self.eps = eps
		self.mu = mu
		self.std = std
	def forward(self, outputs_C, outputs_D, labels):

		A = torch.mul(outputs_C, outputs_D).sum(dim = 1,keepdim = True).repeat(1, args.num_classes) * outputs_C
		B = torch.matmul(outputs_D, self.std)
		Coupling = self.eps*(A + B)

		loss_CE = self.CELoss(outputs_C, labels)

		gt = torch.index_select(self.mu, 0, labels)

		loss_KL = self.alpha * torch.norm(outputs_C - Coupling - gt, dim = 1).mean()

		loss = loss_CE + self.alpha*loss_KL

		return loss
		
def init_setting():
	# init seed
	if args.seed != 0:
		torch.manual_seed(args.seed)

	# Address
	if not os.path.isdir('results'):
		os.mkdir('results')
	if not os.path.isdir('checkpoint'):
		os.mkdir('checkpoint')

	filename = (args.dataset + '_' + args.model + '_' + 'MRL' + '_' +  args.name)
	pth_name = ('checkpoint/' + filename + '.pth')
	log_name = ('results/' + filename + '_LOG.csv')
	detection_name = ('results/' + filename + '_detection.csv')

	#if not os.path.exists(logname):
	with open(log_name, 'w') as logfile:
		logwriter = csv.writer(logfile, delimiter=',')
		logwriter.writerow(['epoch', 'lr', 'train loss', 'train acc', 'test loss', 'test acc'])

	return log_name, pth_name, detection_name

def load_pretrained_model(model):
	PATH = ('./checkpoint/' + args.dataset + '_' + args.model + '_' + 'baseline' + '_' +  args.prename + '.pth')
	print('Loading the pretrained netwokr: ' + PATH)
	net_D = build_model(model, args.num_classes)
	net_D.load_state_dict(torch.load(PATH))
	return net_D

def load_data(dataset):
	mean, std = data_loader.get_known_mean_std(dataset)
	trainloader,testloader = data_loader.load_train_ID(mean, std, dataset, args.batch_size)
	return trainloader, testloader

def adjust_learning_rate(optimizer, epoch):
	if epoch in args.decay_epochs:
		for param_group in optimizer.param_groups:
			new_lr = param_group['lr'] * 0.1
			param_group['lr'] = new_lr

def train(trainloader, net_C, net_D, eps, std, criterion, optimizer):
	net_C.train()
	net_D.eval()

	train_loss = 0
	correct = 0
	total = 0

	for idx, (inputs, targets) in enumerate(trainloader):
		inputs, targets = inputs.cuda(), targets.cuda()

		outputs_C = net_C(inputs)
		outputs_D = net_D(inputs)
		num_classes = outputs_C.size(1)

		loss = criterion(outputs_C, outputs_D, targets)

		train_loss += loss.item()

		A = torch.mul(outputs_C, outputs_D).sum(dim = 1,keepdim = True).repeat(1, num_classes) * outputs_C
		B = torch.matmul(outputs_D, std)
		Coupling = eps*(A + B)

		_, predicted = torch.max(Coupling.data, 1)
		total += targets.size(0)
		correct += (predicted == targets).sum().item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	train_loss = train_loss / idx
	train_acc = 100.*correct/total

	return train_loss, train_acc

def test(testloader, net_C, net_D, eps, std, criterion):
	net_C.eval()
	net_D.eval()

	test_loss = 0
	correct = 0
	total = 0

	with torch.no_grad():
		for idx, (inputs_ID, targets) in enumerate(testloader):
			inputs_ID, targets = inputs_ID.cuda(), targets.cuda()

			outputs_C = net_C(inputs_ID)
			outputs_D = net_D(inputs_ID)

			num_classes = outputs_C.size(1)

			A = torch.mul(outputs_C, outputs_D).sum(dim = 1,keepdim = True).repeat(1, num_classes) * outputs_C
			B = torch.matmul(outputs_D, std)
			Coupling = eps*(A + B)

			loss = criterion(outputs_C, outputs_D, targets)
			test_loss += loss.item() 

			_, predicted1 = torch.max(Coupling.data, 1)

			correct += predicted1.eq(targets).sum().item()

			total += targets.size(0)

	test_loss = test_loss/idx
	test_acc = 100.*correct/total

	return test_loss, test_acc

def detect_OOD_MRL(detection_name, net_C, net_D, testloader, OOD_list):
	print("Measuring the out-of-distribution detection performance.")
	mean, std = data_loader.get_known_mean_std(args.dataset)
	std_D = get_std(testloader, net_D, get_mu(testloader, net_D))

	with open(detection_name, 'w') as logfile:
		logwriter = csv.writer(logfile, delimiter=',')
		logwriter.writerow(['network','in-dist','out-of-dist', 'AUROC', 'AUPR(IN)', 'AUPR(OUT)', 'FPR(95)', 'Detection'])

	soft_ID = OOD_Metric.get_softmax_MRL(net_C, net_D, args.eps, std_D, testloader)

	for dataset_OOD in OOD_list:
		dataloader_OOD = data_loader.load_test_data(mean, std, dataset_OOD)
		soft_OOD = OOD_Metric.get_softmax_MRL(net_C, net_D, args.eps, std_D, dataloader_OOD)
		detection_results = OOD_Metric.detect_OOD(soft_ID, soft_OOD, precision=args.precision)
		with open(detection_name, 'a') as logfile:
			logwriter = csv.writer(logfile, delimiter=',')
			logwriter.writerow([args.model, args.dataset, dataset_OOD,
				detection_results[0], 
				detection_results[1], 
				detection_results[2], 
				detection_results[3], 
				detection_results[4]])

def get_mu(testloader, net):
	net.eval()
	total = 0

	mu = torch.zeros(args.num_classes).cuda()

	with torch.no_grad():
		for idx, (inputs_ID, targets) in enumerate(testloader):

			inputs_ID, targets = inputs_ID.cuda(), targets.cuda()

			outputs_ID = net(inputs_ID)

			mu += torch.mean(outputs_ID, 0)
			total += targets.size(0)

	mu = mu / total
	return mu

def get_std(testloader, net, mu):
	net.eval()
	total = 0

	std = torch.zeros(args.num_classes, args.num_classes).cuda()

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

def save_model(pth_name, net):
	print("Saving: " + pth_name)
	torch.save(net.state_dict(), pth_name)
	print("-------------------------------------------------------------------------")

def save_result(log_name, epoch, train_loss, train_acc, test_loss, test_acc, optimizer):
	with open(log_name, 'a') as logfile:
		logwriter = csv.writer(logfile, delimiter=',')
		logwriter.writerow([epoch, optimizer.state_dict()['param_groups'][0]['lr'],
			train_loss, train_acc, test_loss, test_acc])

def main():

	log_name, pth_name, detection_name = init_setting()

	trainloader, testloader = load_data(dataset = args.dataset)

	net_D = load_pretrained_model(model=args.pretrained_model)
	net_C = build_model(model = args.model, num_classes = args.num_classes)
	
	std_D = get_std(testloader, net_D, get_mu(testloader, net_D))

	mu = torch.zeros(args.num_classes, args.num_classes).cuda()

	criterion = CRLoss(alpha = args.alpha, eps = args.eps, mu = mu, std = std_D)

	optimizer = optim.SGD(net_C.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

	print("Training: " + pth_name)
	for epoch in tqdm(range(0, args.epoch)):
		train_loss, train_acc = train(trainloader, net_C, net_D, args.eps, std_D, criterion, optimizer)
		test_loss, test_acc = test(testloader, net_C, net_D, args.eps, std_D, criterion)
		save_result(log_name, epoch, train_loss, train_acc, test_loss, test_acc, optimizer)
		adjust_learning_rate(optimizer, epoch)
	
	OOD_list = ['LSUN_resize', 'LSUN_crop', 'TinyImageNet_resize', 'TinyImageNet_crop']
	detect_OOD_MRL(detection_name, net_C, net_D, testloader, OOD_list)
	save_model(pth_name, net_C)

if __name__ == '__main__':
	main()
