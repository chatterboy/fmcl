import os
import sys
import time
import argparse
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from data_factory import get_dataset
from model import MyModelForEval
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


def find_threshold(val_loader, model, args):
    model.eval()

    errors = []

    with torch.no_grad():
        for step, (val_data, _) in enumerate(val_loader):
            # val_data [batch size, segment length, # variables]
            val_data = val_data.float().cuda()

            # p, g: [batch size, segment length, # variables]
            p = model(val_data)
            g = val_data

            # pp = []
            # gg = []
            # for idx in range(0, p.size(-1)):
            #     if idx not in [1, 2, 3, 4,
            #                    5, 6, 7, 8, 9,
            #                    10, 11, 12, 13, 14,
            #                    15, 16, 17, 18, 19,
            #                    20, 21, 22, 23, 24,
            #                    25, 26, 27, 28, 29,
            #                    30, 31, 32, 33, 34,
            #                    35, 36, 37, 38, 39,
            #                    40, 41, 42, 43, 44,
            #                    45, 46, 47, 48, 49,
            #                    50]:
            #         pp.append(p[:, :, idx])
            #         gg.append(g[:, :, idx])
            # p = torch.stack(pp, dim=-1)
            # g = torch.stack(gg, dim=-1)

            # e = torch.sum(torch.topk(torch.abs(p - p_m), k=3, dim=1).values, dim=1)
            # e = e / val_data.size(1)
            # e = F.mse_loss(p, g)
            e = torch.sum((p - g) ** 2, dim=(1, 2)) / (p.size(1) * p.size(2))

            errors.append(e)

    errors = torch.cat(errors, dim=0)

    th = torch.quantile(errors, args.quantile)

    return th


def evaluate(test_loader, model, threshold):
    model.eval()

    inputs = []
    recons = []
    errors = []
    targets = []

    with torch.no_grad():
        for step, (test_data, test_labels) in enumerate(test_loader):
            # test_data [batch size, segment length, # variables]
            test_data = test_data.float().cuda()
            test_labels = test_labels.cuda()

            p = model(test_data)
            g = test_data

            # pp = []
            # gg = []
            # for idx in range(0, p.size(-1)):
            #     if idx not in [2, 3, 4,
            #                    5, 6, 7, 8, 9,
            #                    10, 11, 12, 13, 14,
            #                    15, 16, 17, 18, 19,
            #                    20, 21, 22, 23, 24,
            #                    25, 26, 27, 28, 29,
            #                    30, 31, 32, 33, 34,
            #                    35, 36, 37, 38, 39,
            #                    40, 41, 42, 43, 44,
            #                    45, 46, 47, 48, 49,
            #                    50]:
            #         pp.append(p[:, :, idx])
            #         gg.append(g[:, :, idx])
            # p = torch.stack(pp, dim=-1)
            # g = torch.stack(gg, dim=-1)

            e = torch.sum((p - g) ** 2, dim=(1, 2)) / (p.size(1) * p.size(2))

            l = test_labels[:, -1]

            inputs.append(g[:, -1])
            recons.append(p[:, -1])
            errors.append(e)
            targets.append(l)

    inputs = torch.cat(inputs, dim=0)
    recons = torch.cat(recons, dim=0)
    errors = torch.cat(errors, dim=0)
    targets = torch.cat(targets, dim=0)

    inputs = inputs.view(-1)
    recons = recons.view(-1)
    errors = errors.view(-1)
    targets = targets.view(-1)

    preds = (errors > threshold).bool()
    targets = targets.bool()

    preds = point_adjustment(preds, targets)  # TODO:

    # inputs = inputs.detach().cpu().numpy()
    # preds2 = preds2.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    ths = np.array([threshold.item()] * errors.shape[0])
    errors = errors.detach().cpu().numpy()
    x = np.arange(0, preds.shape[0])
    #
    # inputs = inputs[000:30000]
    # preds2 = preds2[25000:30000]
    # preds = preds[25000:30000]
    # targets = targets[:1000]
    # errors = errors[:1000]
    # ths = ths[:1000]
    # x = x[:1000]

    fig, axs = plt.subplots(nrows=2, ncols=1)

    # axs[0].plot(x, preds, label="error")
    axs[0].plot(x, targets, label="target")
    axs[0].legend(loc="lower right")

    axs[1].plot(x, errors, label="errors")
    axs[1].plot(x, ths, label="ths")
    axs[1].legend(loc="lower right")

    plt.show()
    plt.close()

    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    f1 = f1_score(targets, preds)

    return precision, recall, f1


def point_adjustment(p, g):
    anomaly_state = False
    for i in range(g.size(0)):
        if g[i] == 1 and p[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if g[j] == 0:
                    break
                else:
                    if p[j] == 0:
                        p[j] = 1
            for j in range(i, len(g)):
                if g[j] == 0:
                    break
                else:
                    if p[j] == 0:
                        p[j] = 1
        elif g[i] == 0:
            anomaly_state = False
        if anomaly_state:
            p[i] = 1
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save", default="EXP")
    parser.add_argument("--checkpoint", default="finetune-EXP-20230503-123629/init-model.pt")
    parser.add_argument("--dataset", default="SWaT", help="SWaT/WADI/SMAP/MSL")
    parser.add_argument("--data_path", default="data/SWaT")
    parser.add_argument("--window_length", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--quantile", default=0.998, type=float, help="for threshold selection")
    parser.add_argument('--num_features', default=128, type=int)

    args = parser.parse_args()

    args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    os.mkdir(args.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info(args)

    # Load dataset
    train_set, val_set, test_set = get_dataset(args.dataset, args.data_path, args.window_length)

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    # Create a model / load a model
    model = MyModelForEval(num_variables=train_set.data.shape[-1], num_features=args.num_features,
                           window_length=args.window_length)
    model.load_state_dict(torch.load(args.checkpoint))
    model.cuda()

    threshold = find_threshold(val_loader, model, args)
    logging.info("found threshold in validation set %e", threshold)

    precision, recall, f1 = evaluate(test_loader, model, threshold)
    logging.info("precision {} recall {} f1 {}".format(precision, recall, f1))
