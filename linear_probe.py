import os
import sys
import time
import argparse
import logging
from data_factory import get_dataset
from model import MyModel, MyModelForLinearProbing
from loss import ContrastiveLoss
from utils import AverageMeter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim


def train(train_loader, model, optimizer, criterion):
    model.train()

    avg_loss = AverageMeter()

    for step, (train_data, _) in enumerate(train_loader):
        # train_data: [batch size, segment length, # variables] => [B, W, V]
        train_data = train_data.float().cuda()

        ez = model(train_data)

        optimizer.zero_grad()

        loss = criterion(ez, train_data)

        loss.backward()

        optimizer.step()

        avg_loss.update(loss.item(), train_data.size(0))

        # if step % 100 == 0:
        #     logging.info('step %d loss %e', step, avg_loss.avg)

    return avg_loss.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save", default="EXP")
    parser.add_argument("--checkpoint", default="train-EXP-20230428-174816/149-model.pt")
    parser.add_argument("--dataset", default="SWaT", help="SWaT/WADI/SMAP/MSL")
    parser.add_argument("--data_path", default="data/SWaT")
    parser.add_argument("--window_length", default=100, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument('--num_features', default=128, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)  # 5e-5 # 3e-4

    args = parser.parse_args()

    args.save = 'linear-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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
    pretrained = MyModel(num_variables=train_set.data.shape[-1], num_features=args.num_features,
                         window_length=args.window_length)
    pretrained.load_state_dict(torch.load(args.checkpoint))

    model = MyModelForLinearProbing(model=pretrained, num_variables=train_set.data.shape[-1],
                                    num_features=args.num_features)
    model.cuda()

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        loss = train(train_loader, model, optimizer, criterion)
        logging.info('epoch %d loss %e', epoch, loss)

        torch.save(model.state_dict(), os.path.join(args.save, "{}-model.pt".format(epoch)))
