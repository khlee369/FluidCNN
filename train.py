import os
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

from dataset import SampleData
from models.cnn1d import FluidCNN

def main(args):
    train_dataset = SampleData()
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True)

    model = FluidCNN(12, 3)
    model.cuda()

    loss_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr,
                        weight_decay=args.wdecay)

    os.makedirs(args.logs, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logs)

    for epoch in range(1, args.epoch+1):
        for step, [train_x, target_y] in enumerate(tqdm(train_loader)):
            # global_step += 1
            optimizer.zero_grad()
            train_x = train_x.cuda()
            target_y = target_y.cuda()

            preds = model(train_x)
            loss = loss_mse(preds, target_y)

            loss.backward()
            optimizer.step()

        
        train_loss = loss.item()


        print("epoch : ", epoch)
        print("loss : ", train_loss)
        writer.add_scalar('train/Loss', train_loss, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', '--batch_size', type=int, default=8)
    parser.add_argument('-epoch', '--epoch', type=int, default=200)
    parser.add_argument('-lr', '--lr', type=int, default=0.1)
    parser.add_argument('-wdecay', '--wdecay', type=int, default=5e-4)
    parser.add_argument('-momentum', '--momentum', type=int, default=0.9)

    parser.add_argument('-logs', '--logs', type=str, default='logs')

    args = parser.parse_args()

    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    args.logs = os.path.join(args.logs, now)

    cudnn.benchmark = True
    main(args)