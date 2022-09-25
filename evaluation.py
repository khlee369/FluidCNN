import torch
from tqdm import tqdm

def evaluation(model, data_loader):
    total_step = 0
    total_loss = 0
    loss_mse = torch.nn.MSELoss()
    with torch.no_grad():
        for step, [x, y] in enumerate(tqdm(data_loader)):
            total_step+=1

            x = x.cuda()
            y = y.cuda()

            preds = model(x)
            total_loss += loss_mse(preds, y)

    return (total_loss/total_step).item()

if __name__=="__main__":
    from torch.utils.data import DataLoader
    from dataset import FluidData
    from models.cnn1d import FluidCNN
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', '--batch_size', type=int, default=8)
    parser.add_argument('--test_csv', type=str, default='./data/set1/00_CNN_test_Set(s1).csv')
    parser.add_argument('--mpath', type=str, default='./logs/2022_09_25_18_15_24/model_epoch_3.pth')
    args = parser.parse_args()

    test_dataset = FluidData(args.test_csv)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)

    model = FluidCNN(12, 3)
    model.load_state_dict(torch.load(args.mpath))
    model.eval()
    model.cuda()

    print(f"test/Loss : {evaluation(model, test_loader)}")