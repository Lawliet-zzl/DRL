"""
Created on Wed Dec 16 2020
@author: Zhilin Zhao
"""
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from models import *
import data_loader
from model_func import build_model
import OOD_Metric

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--decay_epochs", nargs="+", type=int, default=[50, 100, 150], 
	help="decay learning rate by decay_rate at these epochs")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default="ResNet18", type=str, help='model type (default: ResNet18)')
parser.add_argument('--dataset', default="CIFAR10", type=str, help='dataset')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=20200608, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='total epochs to run')
parser.add_argument('--decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--precision', default=100000, type=float)
parser.add_argument('--num-classes', default=10, type=int, help='the number of classes (default: 10)')
args = parser.parse_args()

args.num_classes = 100 if args.dataset == 'CIFAR100' else 10
best_acc = 0.0

def init_setting():

	# init seed
	if args.seed != 0:
		torch.manual_seed(args.seed)

	# Address
	if not os.path.isdir('results'):
		os.mkdir('results')
	if not os.path.isdir('checkpoint'):
		os.mkdir('checkpoint')

	filename = (args.dataset + '_' + args.model + '_' + 'baseline' + '_' +  args.name)
	pth_name = ('checkpoint/' + filename + '.pth')
	log_name = ('results/' + filename + '_LOG.csv')
	detection_name = ('results/' + filename + '_detection.csv')

	#if not os.path.exists(logname):
	with open(log_name, 'w') as logfile:
		logwriter = csv.writer(logfile, delimiter=',')
		logwriter.writerow(['epoch', 'lr', 'train loss', 'train acc', 'test loss', 'test acc'])

	return log_name, pth_name, detection_name

def load_data(dataset):
	mean, std = data_loader.get_known_mean_std(dataset)
	trainloader,testloader = data_loader.load_train_ID(mean, std, dataset, args.batch_size)
	return trainloader, testloader

def detect_OOD_baseline(detection_name, net, testloader, OOD_list):
	print("Measuring the out-of-distribution detection performance.")
	mean, std = data_loader.get_known_mean_std(args.dataset)
	with open(detection_name, 'w') as logfile:
		logwriter = csv.writer(logfile, delimiter=',')
		logwriter.writerow(['network','in-dist','out-of-dist', 'AUROC', 'AUPR(IN)', 'AUPR(OUT)', 'FPR(95)', 'Detection'])

	soft_ID = OOD_Metric.get_softmax(net, testloader)

	for dataset_OOD in OOD_list:
		dataloader_OOD = data_loader.load_test_data(mean, std, dataset_OOD)
		soft_OOD = OOD_Metric.get_softmax(net, dataloader_OOD)
		detection_results = OOD_Metric.detect_OOD(soft_ID, soft_OOD, precision=args.precision)
		with open(detection_name, 'a') as logfile:
			logwriter = csv.writer(logfile, delimiter=',')
			logwriter.writerow([args.model, args.dataset, dataset_OOD,
				detection_results[0], 
				detection_results[1], 
				detection_results[2], 
				detection_results[3], 
				detection_results[4]])

def train(trainloader, net, criterion, optimizer, epoch):
	net.train()
	train_loss = 0
	correct = 0
	total = 0

	for idx, (inputs, targets) in enumerate(trainloader):
		inputs, targets = inputs.cuda(), targets.cuda()

		outputs = net(inputs)

		loss = criterion(outputs, targets)

		train_loss += loss.item()
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += (predicted == targets).sum().item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	train_loss = train_loss/idx
	train_acc = 100.*correct/total
	return train_loss, train_acc

def test(testloader, net, criterion):
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for idx, (inputs, targets) in enumerate(testloader):
			inputs, targets = inputs.cuda(), targets.cuda()

			outputs = net(inputs)

			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = torch.max(outputs.data, 1)
			total += targets.size(0)

			correct += predicted.eq(targets).sum().item()
	
	test_loss = test_loss/idx
	test_acc = 100.*correct/total

	return test_loss, test_acc

def adjust_learning_rate(optimizer, epoch):
	if epoch in args.decay_epochs:
		for param_group in optimizer.param_groups:
			new_lr = param_group['lr'] * 0.1
			param_group['lr'] = new_lr

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

	net = build_model(model = args.model, num_classes = args.num_classes)

	criterion = nn.CrossEntropyLoss()

	optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

	print("Training: " + pth_name)
	for epoch in tqdm(range(0, args.epoch)):
		train_loss, train_acc = train(trainloader, net, criterion, optimizer, epoch)
		test_loss, test_acc = test(testloader, net, criterion)
		adjust_learning_rate(optimizer, epoch)
		save_result(log_name, epoch, train_loss, train_acc, test_loss, test_acc, optimizer)
	
	OOD_list = ['LSUN_resize', 'LSUN_crop', 'TinyImageNet_resize', 'TinyImageNet_crop']
	detect_OOD_baseline(detection_name, net, testloader, OOD_list)
	save_model(pth_name, net)

if __name__ == '__main__':
	main()

