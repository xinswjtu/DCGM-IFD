import os
import torch
from utils.utils import set_seed, str2bool, Tee, print_environ
import numpy as np
import argparse
from utils.read_cwt_figures import read_directory
from models.classification_models import ResNet18Fc
from torch.utils.data import TensorDataset, DataLoader
import sys
from torch.nn import functional as F

CNNModel = {'ResNet18Fc': ResNet18Fc}

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--fake_train', type=str2bool, default=False, help='是否全部用fake作为训练集')
    parser.add_argument('--is_augment', type=str2bool, default=False, help='是否进行数据增强') # True
    parser.add_argument('--n_augment', type=int, default=15, help='从fake每类取多少个用于增强')
    parser.add_argument('--n_train', type=int, default=30, help='number of training sets')
    parser.add_argument('--logdir', type=str, default=r'logs/Motor/Ours/xx')
    parser.add_argument('--fake_folder', type=str, default='xxx-Eval', help='生成样本所在的文件夹')
    #
    parser.add_argument('--n_test', type=int, default=50, help='test sets from real images')
    parser.add_argument('--epochs', type=int, default=100, help='max number of epoch')
    parser.add_argument('--model', type=str, default='ResNet18Fc')
    parser.add_argument('--real_folder', type=str, default='datasets/Motor/real_images')
    parser.add_argument('--text_label', type=list, help='类别标签')
    parser.add_argument('--n_class', type=int, default=5, help='分类个数')
    parser.add_argument('--img_channel', type=int, default=3, help=' image channel')
    parser.add_argument('--bs', type=int, default=32, help='batch_size of the training process')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--device', type=torch.device, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    args = parser.parse_args()

    args.fake_folder = os.path.join(args.logdir, args.fake_folder, 'fake_images')

    if args.fake_train:
        args.is_augment = False
        args.n_augment = 0

    if args.is_augment:
        args.fake_train = False
    else:
        args.n_augment = 0
    os.makedirs(args.logdir, exist_ok=True)
    return args


def fit(model, loss_fn, optimizer, epochs, train_dl, valid_dl):
    loss_dict = {'train_acc': [], 'valid_acc': [], 'train_loss': [], 'valid_loss': []}
    for epoch in range(epochs):
        model.train()
        train_epoch_loss, train_epoch_acc = 0.0, 0.0
        for xb, yb in train_dl:
            if len(yb.shape) > 1:
                yb = yb.argmax(1)
            xb, yb = xb.to('cuda'), yb.to('cuda')
            output = model(xb)
            loss = loss_fn(output, yb)
            pred = output.argmax(1)
            correct = torch.eq(pred, yb).float().sum().item()
            loss_temp = loss.item() * xb.size(0)
            train_epoch_loss += loss_temp
            train_epoch_acc += correct

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_epoch_loss = train_epoch_loss / len(train_dl.dataset)
        train_epoch_acc = train_epoch_acc / len(train_dl.dataset)
        loss_dict['train_loss'].append(train_epoch_loss)
        loss_dict['train_acc'].append(train_epoch_acc)

        # Validate
        model.eval()
        with torch.no_grad():
            test_epoch_loss, test_epoch_acc = 0.0, 0.0
            for xb, yb in valid_dl:
                if len(yb.shape) > 1:  # 如果yb是one-hot,就转化为一维数组
                    yb = yb.argmax(1)
                xb, yb = xb.to('cuda'), yb.to('cuda')
                output = model(xb)
                loss = loss_fn(output, yb)
                pred = output.argmax(1)
                correct = torch.eq(pred, yb).float().sum().item()
                loss_temp = loss.item() * xb.size(0)
                test_epoch_loss += loss_temp
                test_epoch_acc += correct

            test_epoch_loss = test_epoch_loss / len(valid_dl.dataset)
            test_epoch_acc = test_epoch_acc / len(valid_dl.dataset)
            loss_dict['valid_loss'].append(test_epoch_loss)
            loss_dict['valid_acc'].append(test_epoch_acc)

        print(f"Epoch: {epoch:4d}/{epochs}  Train_acc: {train_epoch_acc:.4f}  Valid_acc: {test_epoch_acc:.4f}  Train_loss: {train_epoch_loss:.4f}  Valid_loss: {test_epoch_loss:.4f}")
    return loss_dict


def model_test(model, target_test_loader):
    model.eval()
    correct = 0
    y_pred, y_true, pred_prob, out_features = [], [], [], []
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to('cuda'), target.to('cuda')
            s_output, fea_out = model(data, features=True) # fea_out for t-SNE
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target).item()
            s_output_prob = F.softmax(s_output, dim=1) # logits --> probability value

            y_pred.append(pred.cpu().numpy())
            y_true.append(target.cpu().numpy())
            pred_prob.append(s_output_prob.cpu())
            out_features.append(fea_out.cpu().numpy())

    acc = correct / len(target_test_loader.dataset)
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    out_features = np.concatenate(out_features, axis=0)  # (n, n_class), float32, numpy

    test_results = {'test_acc': acc, 'y_pred': y_pred, 'y_true': y_true, 'out_features': out_features}
    return test_results

def train(args):
    if args.fake_train:
        train_x, train_y = read_directory(args.fake_folder, args.n_train, args.n_class)
    else:
        train_x, train_y = read_directory(os.path.join(args.real_folder, 'train'), args.n_train, args.n_class)
        if args.is_augment: # 数据扩曾（real+fake）
            fake_x, fake_y = read_directory(args.fake_folder, args.n_augment, args.n_class)
            train_x, train_y = torch.cat((train_x, fake_x), dim=0), torch.cat((train_y, fake_y), dim=0)

    valid_x, valid_y = read_directory(os.path.join(args.real_folder, 'valid'), args.n_test, args.n_class)
    test_x, test_y = read_directory(os.path.join(args.real_folder, 'test'), args.n_test, args.n_class)

    train_ds = TensorDataset(train_x, train_y)
    valid_ds = TensorDataset(valid_x, valid_y)
    test_ds = TensorDataset(test_x, test_y)

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=args.bs)
    test_dl = DataLoader(test_ds, batch_size=args.bs)

    model = CNNModel[args.model](args.img_channel, args.n_class).to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    loss_dict = fit(model, loss_fn, optimizer, args.epochs, train_dl, valid_dl)
    test_results = model_test(model, test_dl)
    return test_results, loss_dict

if __name__ == "__main__":
    set_seed(2023)
    args = parse_args()
    args.text_label = [f"C{i}" for i in range(args.n_class)] # 设置绘制t-SNE的legend label

    sys.stdout = Tee(os.path.join(args.logdir, 'out.txt'))  # 记录输出日志
    print_environ()

    # train
    test_results, loss_dict = train(args)






