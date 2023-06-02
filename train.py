import os
import sys
import time
import argparse
import logging
from data_factory import get_dataset
from model import MyModel
from loss import ContrastiveLoss
from utils import AverageMeter
import torch
from torch.utils.data import DataLoader
import torch.optim as optim


def train(train_loader, model, optimizer, criterion, args):
    model.train()

    avg_loss = AverageMeter()

    for step, (train_data, _) in enumerate(train_loader):
        # train_data: [batch size, segment length, # variables] => [B, W, V]
        train_data = train_data.float().cuda()

        z_1, z_2 = model(train_data)

        optimizer.zero_grad()

        loss = criterion(z_1, z_2)

        loss.backward()

        optimizer.step()

        avg_loss.update(loss.item(), train_data.size(0))

        # if step % 100 == 0:
        #     logging.info('step %d loss %e', step, avg_loss.avg)

    return avg_loss.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='ours')
    parser.add_argument('--num_features', default=128, type=int)
    parser.add_argument('--window_length', default=100, type=int)
    parser.add_argument('--mask_ratio', default=0.5, type=float)
    parser.add_argument('--lr', default=3e-4, type=float)

    parser.add_argument("--save", default="EXP")
    parser.add_argument("--dataset", default="SWaT", help="SWaT/WADI/SMAP/MSL")
    parser.add_argument("--data_path", default="data/SWaT")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_epochs", default=200, type=int)
    parser.add_argument("--temperature", default=0.2, type=float, help="for contrastive loss")

    args = parser.parse_args()

    args.save = 'train-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    os.mkdir(args.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info(args)

    if args.model_name == 'ours':
        # Load dataset
        train_set, val_set, test_set = get_dataset(args.dataset, args.data_path, args.window_length)

        # Create data loaders
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size)
        test_loader = DataLoader(test_set, batch_size=args.batch_size)

        # Create a model / load a model
        model = MyModel(num_variables=train_set.data.shape[-1], num_features=args.num_features,
                        window_length=args.window_length, mask_ratio=args.mask_ratio)
        model = model.cuda()

        criterion = ContrastiveLoss(tau=args.temperature)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.num_epochs):
            loss = train(train_loader, model, optimizer, criterion, args)
            logging.info('epoch %d loss %e', epoch, loss)

            torch.save(model.state_dict(), os.path.join(args.save, "{}-model.pt".format(epoch)))