"""
Created on Nov 18 2021
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
import math
from tqdm import tqdm

from models import *
from model_func import *
from data_loader import load_data
from OOD_Metric import evaluate_detection
from OOD_Metric import detect_OOD

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--decay_epochs", nargs="+", type=int, default=[100, 150], 
	help="decay learning rate by decay_rate at these epochs")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model_D', default="resnet", type=str, help='model type (default: resnet)')
parser.add_argument('--model', default="resnet", type=str, help='model type (default: resnet)')
parser.add_argument('--dataset', default="CIFAR10", type=str, help='dataset')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=19930815, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='total epochs to run')
parser.add_argument('--decay', default=5e-4, type=float, help='weight decay 5e-4')
parser.add_argument('--precision', default=100000, type=float)
parser.add_argument('--num-classes', default=10, type=int, help='the number of classes (default: 10)')
parser.add_argument('--alg', default='DRL', type=str, help='name of algorithm')
parser.add_argument('--save', default=False, action='store_true', help='Save model')
parser.add_argument('--pretrained', default=False, action='store_true', help='load model')
parser.add_argument('--detector', default='ES', type=str, help='detector')
parser.add_argument('--olist', default='A', type=str, help='olist')
parser.add_argument('--test', default=False, action='store_true', help='test')
parser.add_argument('--pth_path', default='checkpoint/', type=str, help='pth_path')
parser.add_argument('--pattern', default='train', type=str, help='pattern')

parser.add_argument('--eps', default=0.01, type=float, help='eps [0.1 0.01 0.001 0.0001], CIFAR10:0.01')
args = parser.parse_args()

class CRLoss(nn.Module):
	"""docstring for CRLoss"""
	def __init__(self, eps = 0.1, std=0.1):
		super(CRLoss, self).__init__()
		self.CELoss = nn.CrossEntropyLoss()
		self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
		self.eps = eps
		self.std = std
	def forward(self, outputs_C, outputs_D, labels):
		C, _, _ = getCoupled(outputs_C, outputs_D, self.eps, self.std)
		loss = self.CELoss(C, labels)
		return loss

def getCoupled(outputs_C, outputs_D, eps, std, temperature = 1):
	num_classes = outputs_C.size(1)
	A = torch.mul(outputs_C, outputs_D).sum(dim = 1,keepdim = True).repeat(1, num_classes) * outputs_C
	B = torch.matmul(outputs_D, std)
	Coupling = eps*(A + B) / args.num_classes
	C = outputs_C - Coupling

	outputs = (C + outputs_D) / 2

	softmax_outputs_C = F.softmax(C.data / temperature, dim=1)
	softmax_outputs_D = F.softmax(outputs_D.data / temperature, dim=1)
	softmax_outputs = (softmax_outputs_C + softmax_outputs_D) / 2

	return C, outputs, softmax_outputs

def train_pretrained(trainloader, net, criterion, optimizer, epoch):
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


def train_DRL(trainloader, net_D, net_C, criterion, optimizer, epoch):
	net_C.train()
	net_D.eval()
	train_loss = 0.0
	correct = 0
	total = 0

	for idx, (inputs, targets) in enumerate(trainloader):
		inputs, targets = inputs.cuda(), targets.cuda()
		outputs_C = net_C(inputs)
		outputs_D = net_D(inputs)
		loss = criterion(outputs_C, outputs_D, targets)
		train_loss += loss.item()
		_, predicted = torch.max(outputs_C.data, 1)
		total += targets.size(0)
		correct += (predicted == targets).sum().item()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	train_loss = train_loss/idx
	train_acc = 100.*correct/total
	return train_loss, train_acc

def test_DRL(testloader, net_D, net_C, eps, std, criterion):
	net_C.eval()
	net_D.eval()

	test_loss = 0
	correct = 0
	test_acc = 0
	total = 0
	with torch.no_grad():
		for idx, (inputs_ID, targets) in enumerate(testloader):
			inputs_ID, targets = inputs_ID.cuda(), targets.cuda()
			outputs_C = net_C(inputs_ID)
			outputs_D = net_D(inputs_ID)
			C, outputs, softmax_outputs = getCoupled(outputs_C, outputs_D, eps, std)
			loss = criterion(outputs, targets)
			test_loss += loss.item()
			_, predicted2 = torch.max( outputs.data, 1)
			correct += predicted2.eq(targets).sum().item()
			total += targets.size(0)

	test_loss = test_loss/idx
	test_acc = 100.*correct/total
	return test_loss, test_acc

def detector_DRL(net_D, net_C, eps, std, dataloader, detector):
	net_C.eval()
	net_D.eval()
	res = np.array([])
	total = 0
	cnt = 0
	with torch.no_grad():
		for idx, (inputs, targets) in enumerate(dataloader):
			inputs= inputs.cuda()
			outputs_C = net_C(inputs)
			outputs_D = net_D(inputs)
			C, Coupling, outputs = getCoupled(outputs_C, outputs_D, eps, std)
			softmax_vals = torch.max(F.softmax(outputs.data, dim=1), dim=1)[0]
			res = np.append(res, softmax_vals.cpu().numpy())
	return res

def Calibrated_DRL(dataloader, net_D, net_C, eps, std, bins=20, temperature = 1, dtype = 'ECE'):
	net_C.eval()
	net_D.eval()
	if dtype == 'ECE':
		ece_criterion = ECELoss_DRL(n_bins=bins).cuda()
	elif dtype == 'AdaECE':
		ece_criterion = AdaECELoss_DRL(n_bins=bins).cuda()
	softmaxes_list = []
	labels_list = []
	with torch.no_grad():
		for idx, (inputs, targets) in enumerate(dataloader):
			inputs= inputs.cuda()
			outputs_C = net_C(inputs)
			outputs_D = net_D(inputs)
			C, outputs, softmax_outputs = getCoupled(outputs_C, outputs_D, eps, std, temperature = temperature)
			softmaxes_list.append(softmax_outputs.cpu())
			labels_list.append(targets.cpu())
		softmaxes = torch.cat(softmaxes_list).cuda()
		labels = torch.cat(labels_list).cuda()
	ece, ys = ece_criterion(softmaxes, labels)
	ece = ece.item()
	return ece, ys

class ECELoss_DRL(nn.Module):
	def __init__(self, n_bins=20):
		super(ECELoss_DRL, self).__init__()
		bin_boundaries = torch.linspace(0, 1, n_bins + 1)
		self.bin_lowers = bin_boundaries[:-1]
		self.bin_uppers = bin_boundaries[1:]
	def forward(self, softmaxes, labels):
		confidences, predictions = torch.max(softmaxes, 1)
		accuracies = predictions.eq(labels)
		ece = torch.zeros(1, device=softmaxes.device)
		ys = []
		for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
			in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
			prop_in_bin = in_bin.float().mean()
			if prop_in_bin.item() > 0:
				accuracy_in_bin = accuracies[in_bin].float().mean()
				avg_confidence_in_bin = confidences[in_bin].mean()
				ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
				accuracy_in_bin = accuracy_in_bin.item()
			else:
				accuracy_in_bin = 0
			ys.append(accuracy_in_bin)
		return ece, ys

class AdaECELoss_DRL(nn.Module):
	def __init__(self, n_bins=20):
		super(AdaECELoss_DRL, self).__init__()
		self.n_bins = n_bins
	def forward(self, softmaxes, labels):
		confidences, predictions = torch.max(softmaxes, 1)
		accuracies = predictions.eq(labels)
		ece = torch.zeros(1, device=softmaxes.device)
		ys = []

		confidences, indices = torch.sort(confidences)
		accuracies = accuracies[indices]

		num = softmaxes.size(0)
		window = int(num / self.n_bins)

		avg_confidence_in_bin = torch.zeros(1, device=softmaxes.device)
		accuracy_in_bin = torch.zeros(1, device=softmaxes.device)
		for i in range(num):
			avg_confidence_in_bin += confidences[i]
			accuracy_in_bin += accuracies[i]
			if (i + 1) % window == 0:
				avg_confidence_in_bin = avg_confidence_in_bin / window
				accuracy_in_bin = accuracy_in_bin / window
				ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * window / num
				ys.append(accuracy_in_bin.item())
				avg_confidence_in_bin = torch.zeros(1, device=softmaxes.device)
				accuracy_in_bin = torch.zeros(1, device=softmaxes.device)
		return ece, ys

def test_detector(dataloader, net_D, net_C, criterion, eps, std):
	net_C.eval()
	net_D.eval()
	res = np.array([])
	with torch.no_grad():
		for idx, (inputs, _) in enumerate(dataloader):
			inputs= inputs.cuda()
			outputs_C = net_C(inputs)
			outputs_D = net_D(inputs)
			C, outputs, softmax_outputs = getCoupled(outputs_C, outputs_D, eps, std)
			softmax_vals = torch.max(softmax_outputs, dim=1)[0]
			res = np.append(res, softmax_vals.cpu().numpy())
	return np.mean(res)

def cal_std(testloader, net_D, num_classes):
	mu = get_mu(testloader, net_D, num_classes)
	std = get_std(testloader, net_D, mu, num_classes)
	return std

def main():
	
	OOD_list = ['CIFAR100','CUB200', 'StanfordDogs120', 'OxfordPets37', 'Oxfordflowers102', 
			'Caltech256', 'DTD47', 'COCO']

	log_name, detection_name, table_name, pth_name = init_setting(args.seed, args.alg, args.dataset, args.model, args.name, args.detector, OOD_list)
	trainloader, testloader = load_data(args.dataset, args.dataset, args.batch_size)
	
	net_D = build_model(args.model_D, args.dataset, args.num_classes)
	net_C = build_model(args.model, args.dataset, args.num_classes)
	
	# load_pretrained(net_D, args.pth_path + 'pretrained_' + args.dataset + '_' + args.model + '.pth')
	pth_name_pre = ('checkpoint/' + 'pretrained_' + args.dataset + '_' + args.model + '_' + args.name + '.pth')
	print("Training the pre-trained network: " + pth_name_pre)
	criterion_CE = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net_C.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
	for epoch in tqdm(range(0, args.epoch)):
		train_loss, train_acc = train_pretrained(trainloader, net_D, criterion_CE, optimizer, epoch)
		adjust_learning_rate(args.decay_epochs, optimizer, epoch)
	save_model(pth_name_pre, net_D, args.save)

	print("Training the auxiliary network: " + pth_name)
	std = cal_std(testloader, net_D, args.num_classes)
	criterion = CRLoss(eps = args.eps, std = std)
	optimizer = optim.SGD(net_C.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
	for epoch in tqdm(range(0, args.epoch)):
		train_loss, train_acc = train_DRL(trainloader, net_D, net_C, criterion, optimizer, epoch)
		test_loss, test_acc = test_DRL(testloader, net_D, net_C, args.eps, std, criterion_CE)
		adjust_learning_rate(args.decay_epochs, optimizer, epoch)
		save_result(log_name, epoch, train_loss, train_acc, test_loss, test_acc, optimizer)
	save_model(pth_name, net_C, args.save)

	print("Testing: " + pth_name + " with " + args.detector)
	ece, ys = Calibrated_DRL(testloader, net_D, net_C, args.eps, std)

	detection_results = np.zeros((len(OOD_list),5))
	soft_ID = detector_DRL(net_D, net_C, args.eps, std, testloader, args.detector)
	for i in tqdm(range(len(OOD_list))):
		dataset_OOD = OOD_list[i]
		testloader_OOD = load_data(args.dataset, dataset_OOD, 100)
		soft_OOD = detector_DRL(net_D, net_C, args.eps, std, testloader_OOD, args.detector)
		detection_results[i,:] = detect_OOD(soft_ID, soft_OOD, precision=args.precision)

	with open(table_name, 'a') as logfile:
		logwriter = csv.writer(logfile, delimiter=',')
		res = [args.alg, args.detector, args.model, args.dataset, args.name,test_acc,ece, ys]
		res.extend([i[0] for i in detection_results])
		logwriter.writerow(res)

	with open(detection_name, 'a') as logfile:
		logwriter = csv.writer(logfile, delimiter=',')
		for i in range(np.size(detection_results,0)):
			logwriter.writerow([OOD_list[i], args.detector,
				detection_results[i][0],
				detection_results[i][1],
				detection_results[i][2],
				detection_results[i][3],
				detection_results[i][4]])

if __name__ == '__main__':
	main()