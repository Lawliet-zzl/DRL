"""
The codes are based on the original codes from https://github.com/ShiyuLiang/odin-pytorch/blob/master/code/calMetric.py
Created on Wed Dec 16 2020
@author: xxx
"""
from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F
from data_loader import load_data
from tqdm import tqdm

def tpr95(soft_IN, soft_OOD, precision):
	#calculate the falsepositive error when tpr is 95%

	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision # precision:200000

	total = 0.0
	fpr = 0.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
		error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		if tpr <= 0.9505 and tpr >= 0.9495:
			fpr += error2
			total += 1
	if total == 0:
		print('corner case')
		fprBase = 1
	else:
		fprBase = fpr/total
	return fprBase

def auroc(soft_IN, soft_OOD, precision):
	#calculate the AUROC
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision
	aurocBase = 0.0
	fprTemp = 1.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
		fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		aurocBase += (-fpr+fprTemp)*tpr
		fprTemp = fpr
	aurocBase += fpr * tpr
	#improve
	return aurocBase

def auroc_XY(soft_IN, soft_OOD, precision):
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision
	aurocBase = 0.0
	fprTemp = 1.0
	tprs = []
	fprs = []
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
		fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		tprs.append(tpr)
		fprs.append(fpr)
	return tprs, fprs

def auprIn(soft_IN, soft_OOD, precision):
	#calculate the AUPR

	precisionVec = []
	recallVec = []
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision

	auprBase = 0.0
	recallTemp = 1.0
	for delta in np.arange(start, end, gap):
		tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
		fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
		if tp + fp == 0: continue
		precision = tp / (tp + fp)
		recall = tp
		precisionVec.append(precision)
		recallVec.append(recall)
		auprBase += (recallTemp-recall)*precision
		recallTemp = recall
	auprBase += recall * precision

	return auprBase

def auprOut(soft_IN, soft_OOD, precision):
	#calculate the AUPR
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision

	auprBase = 0.0
	recallTemp = 1.0
	for delta in np.arange(end, start, -gap):
		fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
		tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
		if tp + fp == 0: break
		precision = tp / (tp + fp)
		recall = tp
		auprBase += (recallTemp-recall)*precision
		recallTemp = recall
	auprBase += recall * precision

	return auprBase

def detection(soft_IN, soft_OOD, precision):
	#calculate the minimum detection error
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision

	errorBase = 1.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
		error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		errorBase = np.minimum(errorBase, (tpr+error2)/2.0)

	return errorBase

def get_softmax_MRL(net_C, net_D, eps, std, dataloader):

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

			num_classes = outputs_C.size(1)
			total += outputs_C.size(0)

			A = torch.mul(outputs_C, outputs_D).sum(dim = 1, keepdim = True).repeat(1, num_classes) * outputs_C
			B = torch.matmul(outputs_D, std)
			Coupling = eps*(A + B) 

			softmax_vals, predicted = torch.max(F.softmax(Coupling.data, dim=1), dim=1)

			res = np.append(res, softmax_vals.cpu().numpy())

	return res
		
def detect_OOD(soft_ID, soft_OOD, precision=200000):
	detection_results = np.array([0.0,0.0,0.0,0.0,0.0])
	detection_results[0] = auroc(soft_ID, soft_OOD, precision)*100
	detection_results[1] = auprIn(soft_ID, soft_OOD, precision)*100
	detection_results[2] = auprOut(soft_ID, soft_OOD, precision)*100
	detection_results[3] = tpr95(soft_ID, soft_OOD, precision)*100
	detection_results[4] = detection(soft_ID, soft_OOD, precision)*100
	return detection_results

def evaluate_detection(params, detector, OOD_list, logname, alg, dataset, trainloader, testloader, net, criterion, precision):
	detection_results = np.zeros((len(OOD_list),5))
	soft_ID = detector(params, net = net, trainloader = trainloader, dataloader = testloader, criterion = criterion)
	for i in tqdm(range(len(OOD_list))):
		dataset_OOD = OOD_list[i]
		testloader_OOD = load_data(dataset, dataset_OOD, 100)
		soft_OOD = detector(params, net = net, trainloader = trainloader, dataloader = testloader_OOD, criterion = criterion)
		detection_results[i,:] = detect_OOD(soft_ID, soft_OOD, precision=precision)
	return detection_results