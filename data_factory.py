import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging


def get_dataset(dataset, data_path, window_size):
    if dataset == "SMD":
        raise NotImplementedError
    elif dataset == "MSL":
        raise NotImplementedError
    elif dataset == "SMAP":
        train_data, val_data, test_data, test_labels = get_SMAP_dataset(data_path)

        logging.info("train data {} val data {} test data {} test labels {}".format(
            train_data.shape, val_data.shape, test_data.shape, test_labels.shape
        ))

        train_fake_labels = np.zeros(train_data.shape)
        val_fake_labels = np.zeros(val_data.shape)

        train_set = MyDataset(train_data, train_fake_labels, window_size)
        val_set = MyDataset(val_data, val_fake_labels, window_size)
        test_set = MyDataset(test_data, test_labels, window_size)

    elif dataset == "SWaT":
        train_data, val_data, test_data, test_labels = get_SWaT_dataset(data_path)

        logging.info("train data {} val data {} test data {} test labels {}".format(
            train_data.shape, val_data.shape, test_data.shape, test_labels.shape
        ))

        train_fake_labels = np.zeros(train_data.shape)
        val_fake_labels = np.zeros(val_data.shape)

        train_set = MyDataset(train_data, train_fake_labels, window_size)
        val_set = MyDataset(val_data, val_fake_labels, window_size)
        test_set = MyDataset(test_data, test_labels, window_size)

    elif dataset == "WADI":
        raise NotImplementedError
    else:
        raise NotImplementedError

    return train_set, val_set, test_set


def get_SMAP_dataset(data_path):
    filename = "A-1.npy"

    train = np.load(os.path.join(data_path, "train", filename))
    test = np.load(os.path.join(data_path, "test", filename))
    labels = np.load(os.path.join(data_path, "labels", filename))

    val = train

    return train, val, test, labels


def get_SWaT_dataset(data_path):
    r = 0.8

    train = np.load(os.path.join(data_path, "train_5_5-10-36.npy"))
    test = np.load(os.path.join(data_path, "test_5_5-10-36.npy"))
    labels = np.load(os.path.join(data_path, "labels_5_5-10-36.npy"))

    # val = train[int(train.shape[0] * r):]
    # train = train[:int(train.shape[0] * r)]
    val = train
    train = train
    test = test
    labels = labels

    return train, val, test, labels


class MyDataset(Dataset):
    def __init__(self, data, labels, window_size):
        self.data = data  # [sequence length, feature dims]
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        return self.data.shape[0] - self.window_size + 1

    def __getitem__(self, idx):
        return self.data[idx:idx + self.window_size], self.labels[idx:idx + self.window_size]
