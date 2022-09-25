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

from dataset import FluidData
from models.cnn1d import FluidCNN
from evaluation import evaluation


def main(args):
    print("Loading train csv...")
    train_dataset = FluidData(args.train_csv)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)

    test_dataset = FluidData(args.test_csv)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)

    print("Init Model...")
    model = FluidCNN(12, 3)
    model.cuda()

    loss_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr,
                        weight_decay=args.wdecay)

    os.makedirs(args.logs, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logs)


    # evluation for debugging
    model.eval()
    test_loss = evaluation(model, test_loader)
    print(f"test/Loss : {test_loss}")
    model.train()

    print("Init Training Loop...")
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
        
        # print loss
        train_loss = loss.item()
        print("epoch : ", epoch)
        print("loss : ", train_loss)

        # evluation
        model.eval()
        test_loss = evaluation(model, test_loader)
        print(f"test/Loss : {test_loss}")
        model.train()

        writer.add_scalar('train/Loss', train_loss, epoch)
        writer.add_scalar('test/Loss', test_loss, epoch)
        torch.save(model.state_dict(), os.path.join(args.logs, f'model_epoch_{epoch}.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', '--batch_size', type=int, default=8)
    parser.add_argument('-epoch', '--epoch', type=int, default=200)
    parser.add_argument('-lr', '--lr', type=int, default=0.1)
    parser.add_argument('-wdecay', '--wdecay', type=int, default=5e-4)
    parser.add_argument('-momentum', '--momentum', type=int, default=0.9)
    parser.add_argument('--train_csv', type=str, default='./data/set1/00_CNN_train_Set(s1).csv')
    parser.add_argument('--test_csv', type=str, default='./data/set1/00_CNN_test_Set(s1).csv')

    parser.add_argument('-logs', '--logs', type=str, default='logs')

    args = parser.parse_args()

    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    args.logs = os.path.join(args.logs, now)

    cudnn.benchmark = True
    main(args)