'''
Author : jayee
Date : 2022-5-20
Function: A toy model used to predict the 'confirmedIncr' based on China-Covid19-province data.

'''

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LSTM, Parameter, GRU, Linear
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_preprocess(args):
    w_size, train_ratio, val_ratio, test_ratio = args.w_size, args.train_ratio, args.val_ratio, args.test_ratio
    file = open('./data/北京市.json', 'r')
    data = json.load(file)['data']
    total_data = []
    for day in data:
        del day['dateId']
        x = torch.as_tensor(list(day.values()), dtype=torch.float)
        total_data.append(x)
    total_data = torch.stack(total_data, dim=0)
    x, y = [], []
    for i in range(total_data.size(0) - w_size):
        x.append(total_data[i:i+w_size, :])
        y.append(total_data[i+w_size, 1]) # 预测每日新增确诊字段'confirmedIncr'
    x = torch.stack(x, dim=0).to(device)
    y = torch.stack(y, dim=0).to(device).view(-1, 1)
    N = x.size(0)
    train_num = int(N * train_ratio)
    val_num = int(N * val_ratio)
    test_num = N - train_num - val_num
    print(f'True Label Rate {(N - test_num)/N}')
    x_train, y_train = x[0:train_num], y[0:train_num]
    x_val, y_val = x[train_num:train_num+val_num], y[train_num:train_num+val_num]
    x_test, y_test = x[train_num+val_num:], y[train_num+val_num:]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

class Covid19(torch.nn.Module):
    def __init__(self, args):
        super(Covid19, self).__init__()
        self.hidden = args.hidden
        self.dropout = args.dropout
        self.LSTM = LSTM(input_size = self.hidden,
                        hidden_size = self.hidden,
                        num_layers = 1,
                        bias = True,
                        batch_first = False,
                        dropout = 0.0,
                        bidirectional = True)
        self.lin = Linear(self.hidden * 2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.LSTM.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.normalize(x, p=2, dim=-1)
        x = x.permute(1, 0, 2)
        _, (h, _) = self.LSTM(x)
        out = torch.cat((h[0], h[1]), dim=-1)
        # print(out.size())
        # out = out.permute(1, 0, 2).squeeze() #(N, hidden)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.lin(out)
        # print(out)
        return out

def train_func(model, optimizer, x, y, args):
    model.train()
    optimizer.zero_grad()
    out = model(x)
    loss_fn = nn.MSELoss()
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
    del out

def test_func(model, x_train, y_train, x_val, y_val, x_test, y_test):
    model.eval()
    loss_fn = nn.MSELoss()
    losses, pres = [], []
    for x, y in [(x_train, y_train), (x_val, y_val), (x_test, y_test)]:
        x = model(x)
        pres.append(x)
        loss = loss_fn(x, y)
        losses.append(loss.detach().cpu())
    return pres, losses

def drawplot(y_hat, y):
    y_hat, y = y_hat.squeeze(), y.squeeze()
    
    # mpl.rcParams['font.sans-serif'] = ['SimHei'] 

    x_axis_data = np.arange(y.size(0))

    plt.plot(x_axis_data, y_hat, color='blue', label='Prediction')
    plt.plot(x_axis_data, y, color='red', label='Ground-truth')

    plt.legend(loc="upper right")
    plt.xlabel('Date')
    plt.ylabel('confirmedIncr')

    plt.show()

    print(y_hat.size(), y.size())


def run(model, train, val, test, args):
    x_train, y_train = train[0], train[1]
    x_val, y_val = val[0], val[1]
    x_test, y_test = test[0], test[1]

    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
    best_val_loss = float('inf')
    test_pre = None
    for epoch in range(args.epochs):
        train_func(model, optimizer, x_train, y_train, args)
        [train_pre, val_pre, tmp_test_pre], [train_loss, val_loss, tmp_test_loss] = test_func(model, 
                                                    x_train, y_train, 
                                                    x_val, y_val, 
                                                    x_test, y_test)
        # if val_loss < best_val_loss:
        #     best_val_pre = val_pre
        #     best_val_loss = val_loss
        test_pre = tmp_test_pre
        print(f'Epoch {epoch} : Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Test Loss = {tmp_test_loss:.4f}')

    y_hat = test_pre.type(torch.int64).cpu()
    y = y_test.cpu()
    drawplot(y_hat, y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--w_size', type=int, default=3)
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--hidden', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=500)

    args = parser.parse_args()

    train, val, test = data_preprocess(args)
    model = Covid19(args).to(device)
    run(model, train, val, test, args)
