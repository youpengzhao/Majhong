import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from changetype import MypklDataset
from dataset import MyDataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from torch.utils.data import ConcatDataset
from tqdm import tqdm
import time,os,gc
import warnings
from memory_profiler import profile
from functools import wraps
warnings.filterwarnings("ignore")

#负责手牌扩充维度
class Net_embedhand(nn.Module):
    def __init__(self):
        super(Net_embedhand, self).__init__()
        self.embed = nn.Embedding(35, 128)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 512)

    def forward(self, x,y):
        x = self.embed(x)
        x = x * y
        x = torch.sum(x, dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        #x = torch.sum(x, dim=0)
        return x

#负责增量牌扩充维度
class Net_embedincre(nn.Module):
    def __init__(self):
        super(Net_embedincre, self).__init__()
        self.embed = nn.Embedding(34, 512)

    def forward(self, x):
        x = self.embed(x)
        x = torch.mean(x, dim=2)
        #x = torch.sum(x, dim=2)
        return x

#负责动作的预测
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1029, 1024)
        #self.bn1 = nn.BatchNorm1d(1)
        self.fc2 = nn.Linear(1024, 512)
        #self.bn2 = nn.BatchNorm1d(1)
        self.fc3 = nn.Linear(512, 256)
        #self.bn3 = nn.BatchNorm1d(1)
        self.fc4 = nn.Linear(256, 256)
        #self.bn4 = nn.BatchNorm1d(1)
        self.fc5 = nn.Linear(256, 235)

    def forward(self, x):
        '''x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))'''

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def testresult(test_loader, PATH1, PATH2, PATH3):       # 这里是调用模型，测试模型在测试集上的准确率
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Net().to(device)
    net_embed1 = Net_embedhand().to(device)
    net_embed2 = Net_embedincre().to(device)

    net_embed1.load_state_dict(torch.load(PATH2, map_location=device))
    net_embed2.load_state_dict(torch.load(PATH3, map_location=device))
    net.load_state_dict(torch.load(PATH1, map_location=device))

    '''net.eval()
    net_embed1.eval()
    net_embed2.eval()'''

    with torch.no_grad():
        acc = []
        L = 10000
        for batch_idx, (handpai, incre, who, act, actlegal) in enumerate(test_loader):
            mul_hand = torch.ge(handpai, 0)
            mul_hand = mul_hand.float()
            mul_hand = mul_hand.unsqueeze(-1)
            mul_hand = mul_hand.to(device)

            hand = torch.clamp(handpai, 0, 34)
            hand=hand.to(device)
            final_h = net_embed1(hand, mul_hand)    # 手牌通过embed层
            final_h = final_h.unsqueeze(1)

            label_i = torch.LongTensor(incre).view(-1, 1)      # 这里是增量牌，具体和手牌一样
            one_hoti = torch.zeros(len(incre), 35, dtype=torch.long).scatter_(1, label_i, 1)
            one_hoti = one_hoti.unsqueeze(1)
            one_hoti = one_hoti.to(device)
            final_i = net_embed2(one_hoti)

            label_w = torch.LongTensor(who).view(-1, 1)
            one_hotw = torch.zeros(len(who), 5, dtype=torch.long).scatter_(1, label_w, 1)
            one_hotw = one_hotw.to(device)
            one_hotw = one_hotw.float()
            one_hotw = one_hotw.unsqueeze(1)               # 增量牌来源，这里为了下面能够将tensor连接起来，需要增添一个维度

            act=act.to(device)                 # 真正的动作
            one_hotl = torch.zeros(len(actlegal), 235, dtype=torch.long).scatter_(1, actlegal, 1)
            one_hotl = one_hotl.to(device)
            one_hotl = one_hotl.float()
            legalact = one_hotl.unsqueeze(1)
            ones=torch.ones_like(legalact)

            data = torch.cat((final_h, final_i), -1)
            data = torch.cat((data, one_hotw), -1)           # 将通过embed的手牌和增量牌以及增量牌来源连接起来
            data = data.to(device)

            res = net(data)
            res = res - (ones.to(device) - legalact.to(device)) * L

            pred = F.softmax(res, dim=-1)              # 预测向量
            loc = torch.argmax(pred, dim=-1)              #选最大的作为预测结果
            com = torch.eq(loc, act)
            com = com.int()
            right=int(com.sum())
            ratio=right/com.numel()
            acc.append(ratio)               # 计算这个batch的准确率
    testacc=np.mean(acc)
    return testacc


def train(epochs, trainfile, trainlegalfile,testfile,testlegalfile,dir):
    PATH1 = './Net_50w.pkl'
    PATH2 = './emhand_50w.pkl'
    PATH3 = './emincre_50w.pkl'
    PATH1m = './Net_50wml.pkl'
    PATH2m = './emhand_50wml.pkl'
    PATH3m = './emincre_50wml.pkl'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Net().to(device)
    net_embed1 = Net_embedhand().to(device)
    net_embed2 = Net_embedincre().to(device)
    BCE_loss = torch.nn.BCELoss().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # 加载模型
    # net_embed1.load_state_dict(torch.load(PATH2, map_location=device))
    # net_embed2.load_state_dict(torch.load(PATH3, map_location=device))
    # net.load_state_dict(torch.load(PATH1, map_location=device))

    L = 10000
    batchsize = 128
    count = 0
    last_loss = 0
    MaxTestAcc = 0

    #这几个数组是画图用的
    loss_pic = []
    trainacc_pic = []
    testacc_pic = []
    testacc_picx = []

    # embedhand_optimizer = optim.Adam(net_embed1.parameters(), lr=0.0003,weight_decay=1e-4)
    # embedincre_optimizer = optim.Adam(net_embed2.parameters(), lr=0.0003,weight_decay=1e-4)
    # net_optimizer = optim.Adam(net.parameters(), lr=0.0003,weight_decay=1e-4)

    #使用一个优化器
    parameters = chain(net.parameters(), net_embed1.parameters(), net_embed2.parameters())
    optimizer = optim.Adam(parameters, lr=0.0005, weight_decay=1e-4)

    #按间距调整学习率
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    '''net.train()
    net_embed1.train()
    net_embed2.train()'''

    firstdir_list=os.listdir(dir)
    firstdir = dir + '/' + firstdir_list[0]
    num = int(len(os.listdir(firstdir)) / 2)

    test_data = MypklDataset(datapkl=testfile, legalpkl=testlegalfile)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batchsize, shuffle=True)  # 加载数据集

    for epoch in range(1, epochs + 1):
        loss_list = []
        right_train=0
        whole_train=0
        time_start = time.time()

        for i in tqdm(range(1,num+1)):
            a = []
            for first in firstdir_list:
                firstdir = dir + '/' + first
                traintxt = firstdir + '/' + trainfile + str(i) + '.pkl'
                trainlegaltxt = firstdir + '/' + trainlegalfile + str(i) + '.pkl'
                train_data = MypklDataset(datapkl=traintxt, legalpkl=trainlegaltxt)
                a.append(train_data)
                del train_data
                gc.collect()
            train_data = ConcatDataset(a)
            train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True)
            for batch_idx, (handpai, incre, who, act, actlegal) in enumerate(train_loader):
                # 使用一个优化器
                optimizer.zero_grad()

                mul_hand = torch.ge(handpai, 0)
                mul_hand = mul_hand.float()
                mul_hand = mul_hand.unsqueeze(-1)
                mul_hand = mul_hand.to(device)
                hand = torch.clamp(handpai, 0, 34)
                # hand=hand.unsqueeze(1)
                hand = hand.to(device)
                final_h = net_embed1(hand, mul_hand)
                final_h = final_h.unsqueeze(1)
                # print(final_h.shape)

                label_i = torch.LongTensor(incre).view(-1, 1)
                one_hoti = torch.zeros(len(incre), 35, dtype=torch.long).scatter_(1, label_i, 1)
                one_hoti = one_hoti.unsqueeze(1)
                one_hoti = one_hoti.to(device)
                final_i = net_embed2(one_hoti)

                label_w = torch.LongTensor(who).view(-1, 1)
                one_hotw = torch.zeros(len(who), 5, dtype=torch.long).scatter_(1, label_w, 1)
                one_hotw = one_hotw.to(device)
                one_hotw = one_hotw.float()
                one_hotw = one_hotw.unsqueeze(1)

                label_a = torch.LongTensor(act)
                one_hota = torch.zeros(len(act), 235, dtype=torch.long).scatter_(1, label_a, 1)
                one_hota = one_hota.unsqueeze(1)
                one_hota = one_hota.float()
                one_hota = one_hota.to(device)

                one_hotl = torch.zeros(len(actlegal), 235, dtype=torch.long).scatter_(1, actlegal, 1)
                one_hotl = one_hotl.float()
                one_hotl = one_hotl.to(device)
                legalact = one_hotl.unsqueeze(1)
                ones = torch.ones_like(legalact)

                data = torch.cat((final_h, final_i), -1)
                data = torch.cat((data, one_hotw), -1)
                data = data.to(device)

                res = net(data)
                res = res - (ones.to(device) - legalact.to(device)) * L
                # print(res.shape)
                pred = F.softmax(res, dim=-1)
                loc = torch.argmax(pred, dim=-1)
                act = act.to(device)
                com = torch.eq(loc, act)
                com = com.int()
                right = int(com.sum())
                right_train += right
                whole_train += com.numel()

                # loss = BCE_loss(pred, one_hota)
                res = res.squeeze(1)
                act = act.squeeze(1)
                loss = criterion(res, act)

                loss_list.append(loss.item())

                loss.backward()
                # embedhand_optimizer.step()      # 更新optimizer
                # embedincre_optimizer.step()
                # net_optimizer.step()

                # 使用一个优化器
                optimizer.step()

                # trainlabel groundtruth
                # print('res:', loc)
                # print('act:', act)
                # if loc != act:
                #     print('handpai:', handpai)
                #     print('actlegal:', actlegal)
            del a
            gc.collect()
            # 按间距调整学习率
            # scheduler.step()
        mean_loss = np.mean(loss_list)
        mean_trainacc = right_train/whole_train
        time_end = time.time()
        time_cost = time_end - time_start
        print(epoch, mean_loss, mean_trainacc, time_cost)
        if abs(last_loss - mean_loss) < 5e-6:
            count += 1
        else:
            count = 0
        loss_pic.append(mean_loss)
        trainacc_pic.append(mean_trainacc)
        if count >= 10:
            break
        if epoch % 4 == 0:  # 每几个epoch保存一下模型，并且测试当前模型的test acc
            torch.save(net.state_dict(), PATH1)
            torch.save(net_embed1.state_dict(), PATH2)
            torch.save(net_embed2.state_dict(), PATH3)
            acc = testresult(test_loader, PATH1, PATH2, PATH3)
            testacc_pic.append(acc)
            testacc_picx.append(epoch)
            print('test acc:', acc)
            if acc > MaxTestAcc:
                torch.save(net.state_dict(), PATH1m)
                torch.save(net_embed1.state_dict(), PATH2m)
                torch.save(net_embed2.state_dict(), PATH3m)
    plt.subplot(3, 1, 1)
    loss_pd = pd.Series(loss_pic)
    loss_pd.plot()
    plt.ylabel('train loss')
    plt.subplot(3, 1, 2)
    plt.plot(testacc_picx, testacc_pic, '.-')
    plt.ylabel('test acc')
    plt.subplot(3, 1, 3)
    trainacc_pd = pd.Series(trainacc_pic)
    trainacc_pd.plot()
    plt.xlabel('epoch')
    plt.ylabel('train acc')
    plt.show()



if __name__ == '__main__':
    epochs = 100
    '''trainfile = './train/train_5w.txt'
    legalfile = './train/trainlegaln_5w.txt'
    testfile = './test/test_5w.txt'
    test_legalfile = './test/testlegaln_5w.txt'''''
    trainfile='train'
    legalfile = 'legaltrain'
    testfile = './test/test.pkl'
    test_legalfile = './test/legaltest.pkl'
    folder='./dividepkl'
    print(torch.cuda.is_available())
    time_start = time.time()
    train(epochs, trainfile, legalfile,testfile,test_legalfile,folder)
    time_end = time.time()
    print('time cost:', time_end - time_start)

    '''for a,b,c in  os.walk('./divide'):
        print(a)
        print(b)
        print(c)'''


    # batchsize = 1024 * 8 * 5
    # print('datatloader start')
    # train_data = MyDataset(datatxt=trainfile, legaltxt=legalfile)
    # print('datatloader finish1')
    # train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True)
    # print('datatloader finish2')
    # # test_data = MyDataset(datatxt=testfile, legaltxt=test_legalfile)
    # # print('datatloader finish3')
    # # test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batchsize, shuffle=True)  # 加载数据集
    # # print('datatloader finish4')
    # PATH1 = './Net.pkl'
    # PATH2 = './emhand.pkl'
    # PATH3 = './emincre.pkl'
    # acc = testresult(train_loader, PATH1, PATH2, PATH3)
    # print('acc: ', acc)