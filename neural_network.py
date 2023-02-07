# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader
from utils import *

class RNN(nn.Module):
	def __init__(self):
		super(self).__init__()
		self.model_input = nn.Sequential(
			nn.LSTM(input_size=1, hidden_size=16, num_layers=2, batch_first=True, dropout=0.1),
		)

		self.model_output = nn.Sequential(
			nn.ReLU(),
			nn.Linear(16, 5)
		)

	def forward(self, x):
		x, _ = self.model_input(x)
		x = self.model_output(x[:, -1, :])
		return x

def as_tensor(X, y, model_type="CNN"):
    X = X.to_numpy()
    y = y.to_numpy()
    data = torch.tensor(X, dtype = torch.float32)
    target = torch.tensor(y, dtype = torch.long).flatten()

    if model_type == 'MLP':
        dataset = TensorDataset(data, target)
    elif model_type == 'CNN':
        data = data.unsqueeze(1)
        dataset = TensorDataset(data, target)
    elif model_type == 'RNN':
        data = data.unsqueeze(2)
        dataset = TensorDataset(data, target)
    else:
        raise AssertionError("Model type not supported.")

    return dataset

if __name__ == "__main__":
    # X, y = get_mushroom_data()
    X, y = get_occupancy_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)

    dataset = as_tensor(X_train, y_train, model_type="MLP")

    MODEL_TYPE = 'RNN'  # TODO: Change this to 'MLP', 'CNN', or 'RNN' according to your task
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    USE_CUDA = False  # Set 'True' if you want to use GPU
    NUM_WORKERS = 0  # Number of threads used by DataLoader. You can adjust this according to your machine spec.



    # X_train = np.array(X_train)
    # y_train = np.array(y_train)
    # X_test = np.array(X_test)
    # y_test = np.array(y_test)

    # model = Sequential()
    # model.add(Dense(4, input_dim=2, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))

    # # Compile model
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # # Train model
    # history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=10)

    # # Plot training and validation accuracy
    # plt.plot(history.history['accuracy'], label='Training accuracy')
    # plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    # plt.legend()
    # plt.show()
# %%