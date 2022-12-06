# https://discuss.pytorch.org/t/cnn-lstm-problem/69344

# https://www.youtube.com/watch?v=0_PgWWmauHk
#   https://github.com/python-engineer/pytorch-examples/blob/master/rnn-lstm-gru/main.py
import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_size, dropout, learning_rate):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.drop_prob = dropout
        self.learning_rate = learning_rate

        # self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5)
        # self.conv2 = torch.nn.Conv1d(in_channels=64,out_channels=64, kernel_size=5)
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # bias=True,
            dropout=dropout,
            batch_first=True,
            bidirectional=False
        )
        # self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x, hidden):
        batch_size, sequence_length, input_dim = x.shape

        #         weight = next(self.parameters()).data
        #         hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda(),
        #                   weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda())

        out_lstm, hidden = self.lstm1(x, hidden)
        # print(out_lstm.size())
        out_view = out_lstm.contiguous().view(batch_size * sequence_length, self.hidden_size)
        # print(out_view.size())
        # out = self.dropout(out)
        # print(out.size())
        out = self.fc(out_view)

        # return the final output and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if(torch.cuda.is_available()):
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda(),
                      weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda())
        else:
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                     weight.new(self.num_layers, batch_size, self.hidden_size).zero_())

        return hidden

class Heuristic(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):

        return out
