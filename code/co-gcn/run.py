from __future__ import print_function

import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import StratifiedShuffleSplit
from util import get_data, construct_graph, sparse_mx_to_torch_sparse_tensor, normalize, load_data
from model import CoGCN
import os


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double().sum().cpu().numpy()
    return correct / len(labels)


def train(args, net, optimizer, feat, targets, lr_alpha, epoch, train_index):
    net.train()
    optimizer.zero_grad()
    output = net.forward(feat)
    loss_train = F.nll_loss(output[train_index], targets[train_index])
    acc_train = accuracy(output[train_index], targets[train_index])
    loss_train.backward()
    optimizer.step()

    # train for an epoch
    net.train()
    output = net.forward(feat)
    loss_train = F.nll_loss(output[train_index], targets[train_index])
    acc_train = accuracy(output[train_index], targets[train_index])
    loss_train.backward()
    net.gc1.update_alpha()
    net.gc2.update_alpha()

    return acc_train


def tes(args, net, feat, targets, epoch, test_index):
    net.eval()
    output = net.forward(feat)
    acc_test = accuracy(output[test_index], targets[test_index])
    return acc_test


def tes_ensemble(args, nets, feats, targets, test_index):
    acc_test = []
    for i in range(len(nets)):
        nets[i].eval()
        output_i = nets[i].forward(feats[i])
        if i == 0:
            output = output_i
        else:
            output = torch.add(output, output_i)
    acc_test = accuracy(output[test_index], targets[test_index])
    return acc_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CoGCN Example')
    parser.add_argument('--mode', type=str, default='train',
                        help='mode of the program')
    parser.add_argument('--dataset', type=str, default='3sources',
                        help='name of the dataset')
    parser.add_argument('--splits', type=int, default=3,
                        help='numbers of data split')
    parser.add_argument('--neighbors', type=int, default=10,
                        help='number of neighbors for graph construction')
    parser.add_argument('--metric', default='euclidean',
                        help='metric for graph construction')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test_batch_size', type=int, default=1024,
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 400)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_alpha', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay rate (default: 5e-4)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed (default: 123)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='number of batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model (default: False)')
    parser.add_argument('--labeled_ratio', type=float, default=0.1,
                        help='Ratio of Labeled Samples (default: 0.05)')
    parser.add_argument('--val_ratio', type=float, default=0.10,
                        help='Ratio of Labeled Samples for Validation (default: 0.10)')
    args = parser.parse_args()

    if not os.path.exists('./models'):
        os.mkdir('./models')
    if not os.path.exists('./param'):
        os.mkdir('./param')

        # Settings
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    name=['Reuters'];
    for name_num in range(len(name)):
        args.dataset = name[name_num]
        file = open('F:\wxh_work\py_semi\co_gcn_master/res/' + args.dataset + '.txt', 'w')
        file.truncate(0)
        fea,Y,num_view,dimension = load_data(args.dataset)
        dimen_total=0
        for i in range(num_view):
            dimen_total=dimen_total+dimension[i]
        sample_num = Y.shape[0]
        feature=np.zeros((sample_num+1,dimen_total))
        cnt=0
        for i in range(num_view):
            feature[0,cnt:cnt+dimension[i]]=i
            feature[1:, cnt:cnt + dimension[i]]=fea[i]
            cnt=cnt+dimension[i]
        target=np.zeros((sample_num,))
        for i in range(sample_num):
            target[i]=Y[i]
        views_ind = feature[0, :]
        n_example = feature.shape[0] - 1
        n_view = int(np.max(views_ind).astype(int) + 1)
        n_class = int(np.max(target).astype(int) + 1)
        n_feat = []
        # Graph Construction
        feats = []
        adjs = []
        for i in range(n_view):
            ind_i = np.arange(feature.shape[1])[views_ind == i]
            feats.append(feature[1:, ind_i])
            n_feat.append(feature[1:, ind_i].shape[1])
            adjs.append(construct_graph(feature[1:, ind_i], target, args.neighbors, args.metric))

        for i in range(n_view):
            feats[i] = torch.FloatTensor(feats[i]).to(device)
            adjs[i] = sparse_mx_to_torch_sparse_tensor(adjs[i]).to(device)
        targets = torch.LongTensor(target).to(device)
        avg_acc = []
        std_acc = []
        for ratio in range(5):
            if ratio==0 and name_num==0:
                continue
            ssv = StratifiedShuffleSplit(n_splits=3, train_size=args.val_ratio, test_size=1.0 - args.val_ratio,
                                         random_state=args.seed)
            ssv.get_n_splits(feature[1:, :], target)

            # Test
            ensemble_accs = []
            for split, (val_index, other_index) in enumerate(ssv.split(feature[1:, :], target)):
                train_index, test_index = [], []
                ssp = StratifiedShuffleSplit(n_splits=1, train_size=(ratio+1)*0.1, test_size=1.0 - (ratio+1)*0.1,
                                             random_state=args.seed)
                ssp.get_n_splits(feature[1:, :][other_index, :], target[other_index])
                for train_ind, test_ind in ssp.split(feature[1:, :][other_index, :], target[other_index]):
                    train_index, test_index = other_index[train_ind], other_index[test_ind]
                nets = []
                optimizers = []
                for i in range(n_view):
                    nets.append(CoGCN(n_feat[i], n_class, adjs, n_view, args.lr_alpha, 0.3, device).to(device))
                    optimizers.append(optim.Adam(nets[i].parameters(), lr=args.lr,
                                                 weight_decay=args.weight_decay))

                for i in range(n_view):
                    best_acc = 0.0
                    # records the coefficients of GC layer
                    gc1 = []
                    gc2 = []
                    for epoch in range(args.epochs):
                        gc1.append(nets[i].gc1.alpha)
                        gc2.append(nets[i].gc2.alpha)

                        train_acc = train(args, nets[i], optimizers[i], feats[i], targets, 0.01, epoch, train_index)
                        val_acc = tes(args, nets[i], feats[i], targets, epoch, val_index)
                        test_acc = tes(args, nets[i], feats[i], targets, epoch, test_index)
                        if val_acc > best_acc:
                            best_acc = val_acc
                            if not os.path.exists('./models/{0}'.format(args.dataset)):
                                os.mkdir('./models/{0}'.format(args.dataset))
                            torch.save(nets[i].state_dict(),
                                       './models/{0}/{1}_view{2}_{3}_{4}_{5}.model'.format(args.dataset, args.metric, i,
                                                                                           args.neighbors, (ratio+1)*0.1,
                                                                                           split))

                    gc1 = np.array(gc1)
                    gc2 = np.array(gc2)
                    if not os.path.exists('./param/{0}'.format(args.dataset)):
                        os.mkdir('./param/{0}'.format(args.dataset))
                    np.savetxt(
                        './param/{0}/{1}_view{2}_{3}_{4}_{5}_gc1.txt'.format(args.dataset, args.metric, i, args.neighbors,
                                                                             (ratio+1)*0.1, split), gc1)
                    np.savetxt(
                        './param/{0}/{1}_view{2}_{3}_{4}_{5}_gc2.txt'.format(args.dataset, args.metric, i, args.neighbors,
                                                                             (ratio+1)*0.1, split), gc2)

                for i in range(n_view):
                    nets[i].load_state_dict(
                        torch.load('./models/{0}/{1}_view{2}_{3}_{4}_{5}.model'.format(args.dataset, args.metric, i,
                                                                                       args.neighbors, (ratio+1)*0.1,
                                                                                       split)))

                ensemble_acc = tes_ensemble(args, nets, feats, targets, test_index)
                ensemble_accs.append(ensemble_acc)
            avg_acc.append(np.mean(ensemble_accs))
            std_acc.append(np.std(ensemble_accs))
        file.write(str(avg_acc))
        file.write('\n')
        file.write(str(std_acc))
        file.write('\n')
        file.close()