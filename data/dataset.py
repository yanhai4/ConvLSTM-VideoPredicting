# -*- coding: utf-8 -*-
# @File : dataset.py
# @Author : 秦博
# @Time : 2022/07/16
import torch
import numpy as np

from torch.utils.data import Dataset


class TrainingSet(Dataset):
    def __init__(self, data_path, length):
        data = np.load(data_path)['sst']
        self.input, self.output = [], []
        for i in range(data.shape[0] - length * 2 + 1):
            self.input.append(data[i:i+length])
            self.output.append(data[i+length:i+length*2])
        self.input = torch.Tensor(np.array(self.input))
        self.output = torch.Tensor(np.array(self.output))

    def __getitem__(self, i):
        return self.input[i], self.output[i]

    def __len__(self):
        return self.input.shape[0]


class ForecastingSet(Dataset):
    def __init__(self, data_path, length):
        data = np.load(data_path)['sst']
        self.input, self.output = [], []
        self.input.append(data[-length:])
        self.output.append(np.zeros(data[-length:].shape))
        self.input = torch.Tensor(np.array(self.input))
        self.output = torch.Tensor(np.array(self.output))

    def __getitem__(self, i):
        return self.input[i], self.output[i]

    def __len__(self):
        return self.input.shape[0]


if __name__ == '__main__':
    data_path = './file/data.npz'
    count = 0
    ds = ForecastingSet(data_path, 3)
    dl = torch.utils.data.DataLoader(dataset=ds, batch_size=1, shuffle=True)
    for i in dl:
        count += 1
    print(count)
