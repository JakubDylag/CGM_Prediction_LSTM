import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import models
import pickle


class LSTM_Trainer():
    def __init__(self, seed, device):
        self.input_size = 0
        self.pred_horizon = 0
        self.batch_size = 0

        self.seed = seed
        self.optimizer, self.loss_func, self.lstm = None, None, None
        self.test_loader, self.train_loader, self.train_dataset, self.test_dataset = None, None, None, None

        torch.manual_seed(seed)
        self.device = device

    def load_train(self, dataset_path, batch_size, input_size, pred_horizon):
        self.batch_size = batch_size
        self.input_size = input_size
        self.pred_horizon = pred_horizon

        [X_train, X_test, y_train, y_test] = pickle.load(open(dataset_path, 'rb'))

        self.train_dataset = SequenceDataset(
            data=X_train,
            target=y_train,
            sequence_length=self.input_size
        )
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def load_test(self, dataset_path, batch_size, input_size, pred_horizon, amount=1000):
        self.batch_size = batch_size
        self.input_size = input_size
        self.pred_horizon = pred_horizon

        [X_train, X_test, y_train, y_test] = pickle.load(open(dataset_path, 'rb'))

        if amount == 0:
            self.test_dataset = SequenceDataset(
                data=X_test,
                target=y_test,
                sequence_length=self.pred_horizon
            )
        else:
            self.test_dataset = SequenceDataset(
                data=X_test[:amount],
                target=y_test[:amount],
                sequence_length=self.pred_horizon
            )

        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    def set_model(self, lstm, loss_func, optimizer):
        self.lstm = lstm
        self.loss_func = loss_func
        self.optimizer = optimizer

    def load_weights(self, model_path):
        checkpoint = torch.load(model_path)
        self.lstm.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

    # Single Epoch Train
    def train(self):
        num_batches = len(self.train_loader)
        total_loss = 0
        self.lstm.train()

        hidden = self.lstm.init_hidden(self.batch_size)
        for X, y in self.train_loader:
            y = y.reshape(y.shape[0] * y.shape[1], y.shape[2])

            # if(X.shape[0] == batch_size):
            output, hidden = self.lstm(X, hidden)
            hidden = tuple([each.data for each in hidden])

            loss = self.loss_func(output.squeeze(), y.squeeze())  # TODO: RMSE with torch.sqrt(criterion(x, y))
            # print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 2)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Train loss: {avg_loss}")

    # TODO: Fix predictions change with different batch_size
    # Test all data in test_loader one-by-one
    def test(self, graph_every):
        num_batches = len(self.test_loader)
        total_loss = 0
        esod_loss = 0

        out_log = []

        self.lstm.eval()
        hidden = self.lstm.init_hidden(self.batch_size)
        prediction_full = torch.zeros([self.batch_size, self.pred_horizon], dtype=torch.float64,
                                      device=self.device)  # store all predictions in pred_horizon
        with torch.no_grad():
            for X, y in self.test_loader:
                for i in range(0, y.shape[1]):
                    prediction_all, hidden = self.lstm(X, hidden)
                    hidden = tuple([each.data for each in hidden])

                    # plt.plot(range(0,self.input_size), X.squeeze())
                    # plt.plot(range(0,self.input_size), prediction_all.squeeze().detach().numpy())
                    # break

                    # get last element (the prediction)
                    prediction_all = torch.reshape(prediction_all, (self.batch_size, self.input_size))
                    prediction = prediction_all[:, -1]

                    # save prediction to list
                    prediction = torch.reshape(prediction, (self.batch_size, 1, 1))  # reshape
                    for j in range(0, self.batch_size):
                        prediction_full[j][i] = prediction[j][0][0].item()  # save prediction to list
                    # print(torch.cat((prediction_full, prediction), dim=0)) TODO: improve efficiency?

                    # add prediction to old X
                    X = torch.cat((X, prediction), 1)  # add prediction to old X
                    X = X[:, 1:, :]

                for a in range(0, len(prediction_full)):
                    out_log.append( [prediction_full[a].cpu().detach().numpy(), y[a].cpu().detach().numpy() ] )

                total_loss += self.loss_func(prediction_full, y).item()

        avg_loss = total_loss / num_batches
        print(f"MSE loss: {avg_loss}")

        return np.array(out_log)

    def test_one(self, graph_every):
        num_batches = len(self.test_loader)
        total_loss = 0

        self.lstm.eval()
        hidden = self.lstm.init_hidden(self.batch_size)
        prediction_full = torch.zeros([1, self.pred_horizon], dtype=torch.float16,
                                      device=self.device)  # store all predictions in pred_horizon
        with torch.no_grad():
            for X, y in self.test_loader:
                for i in range(0, y.shape[1]):
                    prediction_all, hidden = self.lstm(X, hidden)
                    # hidden = tuple([each.data for each in hidden])

                    prediction = prediction_all[-1][0]  # get last element (the prediction)
                    prediction = torch.reshape(prediction, (1, 1))  # reshape

                    prediction_full[0][i] = prediction[0][0].item()  # save prediction to list

                    X = torch.cat((X[0][1:self.input_size], prediction), 0)  # add prediction to old X
                    X = torch.reshape(X, (1, self.input_size, 1))  # reshape

                total_loss += self.loss_func(prediction_full, y).item()

        avg_loss = total_loss / num_batches
        print(f"Test loss: {avg_loss}")

    def clear(self):
        print(torch.cuda.memory_allocated())
        print(torch.cuda.memory_reserved())

        del self.train_loader
        del self.train_dataset
        del self.test_loader
        del self.test_dataset
        del self.lstm
        torch.cuda.empty_cache()

        print(torch.cuda.memory_allocated())
        print(torch.cuda.memory_reserved())


# https://www.crosstab.io/articles/time-series-pytorch-lstm#:~:text=Create%20datasets%20that%20PyTorch%20DataLoader%20can%20work%20with
class SequenceDataset(Dataset):
    def __init__(self, data, target, sequence_length=5, device='cuda'):
        self.sequence_length = sequence_length
        self.y = torch.tensor(target).float()
        self.y = self.y.to(device)
        self.X = torch.tensor(data).float()
        self.X = self.X.to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        #         if i >= self.sequence_length - 1:
        #             i_start = i - self.sequence_length + 1
        #             x = self.X[i_start:(i + 1), :]
        #         else:
        #             padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
        #             x = self.X[0:(i + 1), :]
        #             x = torch.cat((padding, x), 0) #FIX

        return self.X[i], self.y[i]
