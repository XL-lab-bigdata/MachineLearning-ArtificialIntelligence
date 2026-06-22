import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# 设置随机种子以确保可重复性
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 参数解析函数
def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning with FedAvg')

    # 联邦学习参数
    parser.add_argument('--epochs', type=int, default=300, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum")

    # 模型与数据集参数
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--dataset', type=str, default='FashionMNIST', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of images")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--verbose', action='store_true', help='verbose printing')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')

    return parser.parse_args()


# 数据集划分工具类
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


# 数据划分函数
def FashionMNIST_iid(dataset, num_users):
    """将FashionMNIST数据集按照IID方式划分给客户端"""
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users


def FashionMNIST_noniid(dataset, num_users):
    """将FashionMNIST数据集按照Non-IID方式划分给客户端"""
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)

    # 获取标签并排序
    labels = dataset.targets.numpy()
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 按标签排序
    idxs = idxs_labels[0, :]

    # 分配数据
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)

        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    return dict_users


# 模型定义
class MLP(nn.Module):
    """多层感知机模型"""

    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNFashionMNIST(nn.Module):
    """CNN模型用于FashionMNIST分类"""

    def __init__(self, args):
        super(CNNFashionMNIST, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


# 本地训练类
class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs),
            batch_size=self.args.local_bs,
            shuffle=True
        )

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum
        )

        epoch_loss = []

        for _ in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        _, batch_idx * len(images), len(self.ldr_train.dataset),
                           100. * batch_idx / len(self.ldr_train), loss.item()))

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


# 联邦平均算法
def FedAvg(w):
    """联邦平均算法实现"""
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg


# 测试函数
def test_img(net, dataset, args):
    net.eval()
    test_loader = DataLoader(dataset, batch_size=args.bs, shuffle=False)
    loss_total = 0
    correct = 0
    loss_func = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = net(data)
            loss = loss_func(output, target)
            loss_total += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    accuracy = 100.0 * correct / len(test_loader.dataset)
    avg_loss = loss_total / len(test_loader)

    return accuracy, avg_loss


# 主函数
def main():
    # 解析参数
    args = parse_args()

    # 设置设备
    args.device = torch.device('cuda:{}'.format(args.gpu)
                               if torch.cuda.is_available() and args.gpu != -1
                               else 'cpu')

    # 设置随机种子
    set_seed(args.seed)

    # 加载数据集
    if args.dataset == 'FashionMNIST':
        trans_FashionMNIST = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_train = datasets.FashionMNIST(
            '../data/FashionMNIST/',
            train=True,
            download=True,
            transform=trans_FashionMNIST
        )
        dataset_test = datasets.FashionMNIST(
            '../data/FashionMNIST/',
            train=False,
            download=True,
            transform=trans_FashionMNIST
        )

        # 数据划分
        if args.iid:
            dict_users = FashionMNIST_iid(dataset_train, args.num_users)
        else:
            dict_users = FashionMNIST_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')

    # 获取图像尺寸
    img_size = dataset_train[0][0].shape

    # 构建模型
    if args.model == 'cnn' and args.dataset == 'FashionMNIST':
        net_glob = CNNFashionMNIST(args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    print(net_glob)
    net_glob.train()

    # 复制权重
    w_glob = net_glob.state_dict()

    # 训练初始化
    loss_train = []

    # 主训练循环
    for iter in range(args.epochs):
        loss_locals = []

        if not args.all_clients:
            w_locals = []

        # 选择客户端
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # 本地训练
        for idx in idxs_users:
            local = LocalUpdate(args, dataset_train, dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))

            loss_locals.append(copy.deepcopy(loss))

        # 聚合权重
        w_glob = FedAvg(w_locals)

        # 更新全局模型
        net_glob.load_state_dict(w_glob)

        # 打印损失
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # 测试
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Training accuracy: {:.2f}%".format(acc_train))
    print("Testing accuracy: {:.2f}%".format(acc_test))


if __name__ == '__main__':
    main()