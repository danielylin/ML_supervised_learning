# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader
from utils import *
import time


class MLP(nn.Module):
	def __init__(self, in_features):
		super().__init__()

		self.model = nn.Sequential(
			nn.Linear(in_features=in_features, out_features=16),
			nn.Dropout(p=0.3),
			nn.ReLU(),
			# nn.Dropout(p=0.3),
			# nn.ReLU(),
			nn.Linear(in_features=16, out_features=5)
		)

	def forward(self, x):
		x = self.model(x)
		return x

def format_mlp_tensor(X, y):
    X = X.to_numpy()
    y = y.to_numpy()
    X_tensor = torch.tensor(X, dtype = torch.float32)
    y_tensor = torch.tensor(y, dtype = torch.long).flatten()

    data = TensorDataset(X_tensor, y_tensor)

    return data

def train_nn(model, train_loader, data_loader, criterion, optimizer, epoch, print_freq=10):
	model.train()
	for i, (input, target) in enumerate(data_loader):
		# measure data loading time

		input = input.to(device)
		target = target.to(device)

		optimizer.zero_grad()
		output = model(input)
		loss = criterion(output, target)
		assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

		loss.backward()
		optimizer.step()

		# measure elapsed time

		losses.update(loss.item(), target.size(0))
		accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

		if i % print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
				epoch, i, len(data_loader), batch_time=batch_time,
				data_time=data_time, loss=losses, acc=accuracy))

	return losses.avg, accuracy.avg

def evaluate(model, device, data_loader, criterion, print_freq=10):
	batch_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()

	results = []

	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (input, target) in enumerate(data_loader):

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			target = target.to(device)

			output = model(input)
			loss = criterion(output, target)

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			losses.update(loss.item(), target.size(0))
			accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

			y_true = target.detach().to('cpu').numpy().tolist()
			y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
			results.extend(list(zip(y_true, y_pred)))

			if i % print_freq == 0:
				print('Test: [{0}/{1}]\t'
					#   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
					i, len(data_loader), loss=losses, acc=accuracy))

	return losses.avg, accuracy.avg, results

if __name__ == "__main__":
	X, y = get_mushroom_data()
	# X, y = get_occupancy_data()
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.30, random_state=42)

	train_dataset = format_mlp_tensor(X_train, y_train)
	test_dataset = format_mlp_tensor(X_test, y_test)
	input_tensor, _ = train_dataset[0]
	input_features = input_tensor.numel()

	epoch = 15
	batch_size = 32
	num_workers = 0

	for epoch in range(NUM_EPOCHS):
		device = torch.device("cpu")
		torch.manual_seed(1)
		train_data = torch.utils.data.DataLoader(train_dataset,
													batch_size=batch_size,
													shuffle=True,
													num_workers=num_workers)

		test_data = torch.utils.data.DataLoader(test_dataset,
												batch_size=batch_size,
												shuffle=True,
												num_workers=num_workers)

		train_loss_list, train_accuracy_list = [], []
		test_loss_list, test_accuracy_list = [], []

		model = MLP(in_features=input_features)

		# Define a criterion and optimizer.
		criterion = nn.CrossEntropyLoss()
		# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
		optimizer = optim.Adam(model.parameters())

		train_loss, train_accuracy = train_nn(model, device, train_data, criterion, optimizer, epoch)
		test_loss, test_accuracy, test_res = evaluate(model, device, test_data, criterion)
		train_loss_list.append(train_loss)
		test_loss_list.append(test_loss)
		train_accuracy_list.append(train_accuracy)
		test_accuracy_list.append(test_accuracy)

		print(train_loss_list)

# %%