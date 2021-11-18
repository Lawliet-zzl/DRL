"""
Created on Wed Dec 16 2020
@author: xxx
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
from PIL import Image
import os
import sys
import os.path
import numpy as np
import math
import random
from numpy.random import randint

class LoadDataset(Dataset):
	"""docstring for LoadDataset"""
	def __init__(self,  root, list_file='train', transform=None, target_transform=None, full_dir=True):
		super(LoadDataset, self).__init__()
		self.root = root
		self.list_file = list_file
		self.transform = transform
		self.target_transform = target_transform
		self.full_dir = full_dir
		self._parse_list()

	def _load_image(self, directory):
		if self.full_dir:
			return Image.open(directory).convert('RGB')
		else:
			return Image.open(os.path.join(self.root, 'data', directory)).convert('RGB')

	def _parse_list(self):
		self.data_list = [LoadRecord(x.strip().split(' ')) for x in open(os.path.join(self.root, self.list_file))]

	def __getitem__(self, index):
		record = self.data_list[index]

		return self.get(record)

	def get(self, record, indices=None):
		img = self._load_image(record.path)

		process_data = self.transform(img)
		if not self.target_transform == None:
			process_label = self.target_transform(record.label)
		else:
			process_label = record.label

		return process_data, process_label

	def __len__(self):
		return len(self.data_list)

class LoadRecord(object):
	"""docstring for LoadRecord"""
	def __init__(self, data):
		super(LoadRecord, self).__init__()
		self._data = data

	@property
	def path(self):
		return self._data[0]

	@property
	def label(self):
		return int(self._data[1])

def getCIFAR10(transform, batch_size=128, shuffle=True, train=True, test=True):

	ds = []

	if train:
		trainset = datasets.CIFAR10(root='../data/CIFAR10', train=True, download=True, transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
		ds.append(trainloader)
	if test:
		testset = datasets.CIFAR10(root='../data/CIFAR10', train=False, download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
		ds.append(testloader)

	ds = ds[0] if len(ds) == 1 else ds
	return ds

def getCIFAR100(transform, batch_size=128, shuffle=True, train=True, test=True):

	ds = []

	if train:
		trainset = datasets.CIFAR100(root='../data/CIFAR100', train=True, download=True, transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
		ds.append(trainloader)
	if test:
		testset = datasets.CIFAR100(root='../data/CIFAR100', train=False, download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
		ds.append(testloader)

	ds = ds[0] if len(ds) == 1 else ds
	return ds

def getAllOOD(root, transform, batch_size=128, shuffle=False, glist=False):
	if glist:
		generate_all_list(root=root)
	dataset = LoadDataset(root=root, list_file='all_list.txt', transform=transform, full_dir=False)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
	return dataloader

def getCOCO(transform, batch_size=128, shuffle=False):
	dataset = datasets.ImageFolder('../data/COCO',transform=transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=8)
	return dataloader

def get_known_mean_std(dataset):
	if dataset == 'CIFAR10':
		mean = (0.4914, 0.4822, 0.4465)
		std = (0.2470, 0.2435, 0.2616)
	return mean, std
		
def get_transform(dataset):
	mean = (0.4914, 0.4822, 0.4465)
	std = (0.2470, 0.2435, 0.2616)
	transform_train = transforms.Compose([
		transforms.Resize(32),
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
		])
	transform_test = transforms.Compose([
		transforms.Resize((32,32)),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
		])
	return transform_train, transform_test

def load_train_ID(transform_train, transform_test, dataset, batch_size):
	trainloader = getCIFAR10(transform_train, batch_size=batch_size, shuffle=True, train=True, test=False)
	testloader = getCIFAR10(transform_test, batch_size=100, shuffle=False, train=False, test=True)
	return trainloader, testloader

def load_test_OOD(transform_test, dataset):
	if dataset == 'CIFAR100':
		testloader = getCIFAR100(transform_test, batch_size=100, shuffle=False, train=False, test=True)
	elif dataset == 'COCO':
		testloader = getCOCO(transform_test, batch_size=100, shuffle=False)
	else:
		testloader = getAllOOD(root = '../data/' + dataset + '/',transform = transform_test, 
			batch_size=100, shuffle=False, glist=False)
	return testloader

def load_data(dataset_ID, dataset, batch_size):
	transform_train, transform_test = get_transform(dataset_ID)
	if dataset_ID == dataset:
		trainloader,testloader = load_train_ID(transform_train, transform_test, dataset, batch_size)
		return trainloader, testloader
	else:
		testloader = load_test_OOD(transform_test, dataset)
		return testloader

