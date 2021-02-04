"""
Created on Wed Dec 16 2020
@author: Zhilin Zhao
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
from numpy.random import randint

def getCIFAR10(mean, std, batch_size=128, shuffle=True, train=True, test=True):

	transform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
		])

	ds = []

	if train:
		print("Loading CIFAR10 training dataset (in-distribution)")
		trainset = datasets.CIFAR10(root='../data_RCL/CIFAR10', train=True, download=True, transform=transform)

		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
		ds.append(trainloader)
	if test:
		print("Loading CIFAR10 testing dataset (in-distribution)")
		testset = datasets.CIFAR10(root='../data_RCL/CIFAR10', train=False, download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
		ds.append(testloader)

	ds = ds[0] if len(ds) == 1 else ds
	return ds

def getCIFAR100(mean, std, batch_size=128, shuffle=True, train=True, test=True):

	transform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
		])

	ds = []

	if train:
		print("Loading CIFAR100 training dataset (in-distribution)")
		trainset = datasets.CIFAR100(root='../data_RCL/CIFAR100', train=True, download=True, transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
		ds.append(trainloader)
	if test:
		print("Loading CIFAR100 testing dataset (in-distribution)")
		testset = datasets.CIFAR100(root='../data_RCL/CIFAR100', train=False, download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
		ds.append(testloader)

	ds = ds[0] if len(ds) == 1 else ds
	return ds

def getLSUN(mean, std, version, batch_size=128, shuffle=True):
	transform = transforms.Compose([
	transforms.Resize((32,32)),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])
	if version == 'crop':
		print("Loading LSUN (c) dataset (out-of-distribution)")
		dataset = datasets.ImageFolder('../data_RCL/LSUN_crop',transform=transform)
	elif version == 'resize':
		print("Loading LSUN (r) dataset (out-of-distribution)")
		dataset = datasets.ImageFolder('../data_RCL/LSUN_resize',transform=transform)

	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=9)
	return dataloader

def getTinyImagenet(mean, std, version, batch_size=128, shuffle=True):
	transform = transforms.Compose([
	transforms.Resize((32,32)),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])
	if version == 'crop':
		print("Loading TinyImageNet (c) dataset (out-of-distribution)")
		dataset = datasets.ImageFolder('../data_RCL/TinyImageNet_crop',transform=transform)
	elif version == 'resize':
		print("Loading TinyImageNet (r) dataset (out-of-distribution)")
		dataset = datasets.ImageFolder('../data_RCL/TinyImageNet_resize',transform=transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=9)
	return dataloader

def get_known_mean_std(dataset):
	if dataset == 'CIFAR10':
		mean = (0.4914, 0.4822, 0.4465)
		std = (0.2470, 0.2435, 0.2616)
	elif dataset == 'CIFAR100':
		mean = (0.5071, 0.4865, 0.4409)
		std = (0.2673, 0.2564, 0.2762)
	return mean, std

def load_test_data(mean, std, dataset, batch_size = 100):
	
	if dataset == 'LSUN_resize':
		return getLSUN(mean, std, version='resize', batch_size=batch_size, shuffle=False)
	elif dataset == 'LSUN_crop':
		return getLSUN(mean, std, version='crop', batch_size=batch_size, shuffle=False)
	elif dataset == 'TinyImageNet_resize':
		return getTinyImagenet(mean, std, version='resize', batch_size=batch_size, shuffle=False)
	elif dataset == 'TinyImageNet_crop':
		return getTinyImagenet(mean, std, version='crop', batch_size=batch_size, shuffle=False)
		
def load_train_ID(mean, std, dataset, batch_size):

	if dataset == 'CIFAR10':
		trainloader = getCIFAR10(mean, std, batch_size=batch_size, shuffle=True, train=True, test=False)
		testloader = getCIFAR10(mean, std, batch_size=100, shuffle=False, train=False, test=True)
	elif dataset == 'CIFAR100':
		trainloader = getCIFAR100(mean, std, batch_size=batch_size, shuffle=True, train=True, test=False)
		testloader = getCIFAR100(mean, std, batch_size=100, shuffle=False, train=False, test=True)
	return trainloader, testloader



