import torch
import numpy as np
import os,time,math,fileinput
from torch.utils import data
from try1 import getdata,writefile,get_legal
from legal_action_judge import legal_action_judge1,legal_action_judge2
from randaction import myjudge
import time, random


class createDataset(data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, datatxt,legaltxt):  # 初始化一些需要传入的参数
        super(createDataset, self).__init__()
        dic=getdata(datatxt)
        self.hand = dic['handpai']
        self.incre= dic['increment']
        self.who= dic['who']
        self.action=dic['act']
        self.legal=myjudge(legaltxt)

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        handpai = self.hand[index]
        incre = self.incre[index]
        who = self.who[index]
        act = self.action[index]
        actlegal=self.legal[index]

        return handpai,incre,who,act,actlegal# return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.hand)


class MyDataset(data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, datatxt, legaltxt):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        dic = getdata(datatxt)
        self.hand = dic['handpai']
        self.incre = dic['increment']
        self.who = dic['who']
        self.action = dic['act']
        self.legal = get_legal(legaltxt)

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        handpai = self.hand[index]
        while len(handpai)<16:
            handpai.append(-1)
        handpai=torch.as_tensor(handpai)
        incre = self.incre[index]
        incre=torch.as_tensor(incre)
        who = self.who[index]
        who=torch.as_tensor(who)
        actnum = self.action[index]
        act=[]
        act.append(actnum)
        act=torch.as_tensor(act)
        actlegal = self.legal[index]
        actlegal=torch.as_tensor(actlegal)

        return handpai, incre, who, act, actlegal  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.hand)


def write_delete(s1,s2,domin1,domin2):
    filedir1 = s1
    filedir2 = s2
    # 获取当前文件夹中的文件名称列表
    filenames = os.listdir(filedir2)
    length = len(filenames)
    batch = 100
    n = math.ceil(length / batch)
    num = [i for i in range(1, n + 1)]
    i = 0
    count = 0
    state_train = domin2 + 'train' + str(num[i]) + ".txt"
    tezheng_train = domin1 + 'train' + str(num[i]) + '.txt'
    state_test = domin2 + 'test' + str(num[i]) + ".txt"
    tezheng_test = domin1 + 'test' + str(num[i]) + '.txt'
    fstate_train = open(state_train, 'w')
    ftezheng_train = open(tezheng_train, 'w')
    fstate_test = open(state_test, 'w')
    ftezheng_test = open(tezheng_test, 'w')
    # 打开当前目录下的result.txt文件，如果没有则创建
    max = 15
    for filename in filenames:
        filepath2 = filedir2 + '/' + filename
        filepath1 = filedir1 + '/' + filename
        # print(filepath)
        # 遍历单个文件，读取行数
        with open(filepath1, 'r') as f1, open(filepath2, 'r') as f2:
            for line1, line2 in zip(f1, f2):
                if eval(line2)[0][9][0] == 4:
                    line = eval(line2)
                    act_legal = legal_action_judge2(line)
                    # act_legal =legaltransform(act_legal)
                else:
                    line = eval(line2)
                    act_legal = legal_action_judge1(line)
                num1 = act_legal.count(1)
                if num1 == 1:
                    continue
                reslegal = [i for i, x in enumerate(act_legal) if x == 1]
                while len(reslegal) < max:
                    reslegal.append(reslegal[0])
                reslegal = str(reslegal) + '\n'
                x = random.random()
                if x <= 0.8:
                    ftezheng_train.writelines(line1)
                    fstate_train.writelines(reslegal)
                else:
                    ftezheng_test.writelines(line1)
                    fstate_test.writelines(reslegal)
        count += 1
        if count % batch == 0:
            fstate_train.close()
            ftezheng_train.close()
            fstate_test.close()
            ftezheng_test.close()
            i += 1
            if count != length:
                state_train = domin2 + 'train' + str(num[i]) + ".txt"
                tezheng_train = domin1 + 'train' + str(num[i]) + '.txt'
                state_test = domin2 + 'test' + str(num[i]) + ".txt"
                tezheng_test = domin1 + 'test' + str(num[i]) + '.txt'
                fstate_train = open(state_train, 'w')
                ftezheng_train = open(tezheng_train, 'w')
                fstate_test = open(state_test, 'w')
                ftezheng_test = open(tezheng_test, 'w')
    fstate_train.close()
    ftezheng_train.close()
    fstate_test.close()
    ftezheng_test.close()
    '''for line in fileinput.input(filepath2):
            f2.writelines(line)
        f2.write('\n')
        count += 1
        if count % 100 == 0:
            f2.close()
            i += 1
            if count != length:
                state = domin2 + str(num[i])+'.txt'
                print(state)
                f2 = open(state, 'w')
    f2.close()
    i=0
    count=0
    tezheng = domin1 + str(num[i])+'.txt'
    f = open(tezheng, 'w')
    for filename in filenames:
        filepath1 = filedir1 + '/' + filename
        # print(filepath)
        # 遍历单个文件，读取行数
        for line in open(filepath1):
            f.writelines(line)
        f.write('\n')
        count += 1
        if count % 100 == 0:
            f.close()
            i += 1
            if count!=length:
                tezheng = domin1 + str(num[i])+'.txt'
                f = open(tezheng, 'w')
    f.close()'''


def changestate(state):
    a = str(state[0][0]).replace('(', '').replace(',)', '')
    state[0][0] = eval(a)
    a = str(state[0][1]).replace('(', '').replace(',)', '')
    state[0][1] = eval(a)
    a = str(state[0][2]).replace('(', '').replace(',)', '')
    state[0][2] = eval(a)
    a = str(state[0][3]).replace('(', '').replace(',)', '')
    state[0][3] = eval(a)
    a = str(state[0][4]).replace('(', '').replace(',)', '')
    state[0][4] = eval(a)
    a = str(state[0][5]).replace('(', '').replace(',)', '')
    state[0][5] = eval(a)
    a = str(state[0][6]).replace('(', '').replace(',)', '')
    state[0][6] = eval(a)
    a = str(state[0][7]).replace('(', '').replace(',)', '')
    state[0][7] = eval(a)
    a = str(state[0][8]).replace('(', '').replace(',)', '')
    state[0][8] = eval(a)
    state[0][9] = list(map(int, state[0][9]))
    state[1] = int(state[1])

    return state

# 根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器

def dividedataset(datatxt,legaltxt,trainfile,legalfile,testfile,test_legalfile):
    datatxt=datatxt
    legaltxt=legaltxt
    totaldata = createDataset(datatxt,legaltxt)
    test_split=0.2
    shuffle_dataset=True
    random_seed=42
    datasize=len(totaldata)
    indices=list(range(datasize))
    split=int(np.floor(test_split*datasize))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices,test_indices=indices[split:],indices[:split]
    train_sampler=data.SubsetRandomSampler(train_indices)
    test_sampler = data.SubsetRandomSampler(test_indices)
    trainloader = data.DataLoader(dataset=totaldata, batch_size=1, sampler=train_sampler)
    testloader = data.DataLoader(dataset=totaldata, batch_size=1, sampler=test_sampler)
    f_train = open(trainfile,'w+',encoding='UTF-8')
    flegal_train = open(legalfile,'w+',encoding='UTF-8')
    max = 15

    for batch_idx, (handpai, incre, who, act, actlegal) in enumerate(trainloader):
        hand = list(map(int, handpai))
        incre = list(map(int, incre))
        who = list(map(int, who))
        all = who + incre
        act = int(act)
        res = []
        res = '[' + str(hand) + ',' + str(all) + ',' + str(act) + ']' + '\n'
        # print(res)
        reslegal = list(map(int, actlegal))
        num1 = reslegal.count(1)
        if num1 == 1:
            continue
        reslegal = [i for i, x in enumerate(reslegal) if x == 1]
        while len(reslegal) < max:
            reslegal.append(reslegal[0])
        reslegal = str(reslegal) + '\n'
        f_train.write(res)
        flegal_train.write(reslegal)

    f_train.close()
    flegal_train.close()
    f_test = open(testfile, 'w+', encoding='UTF-8')
    flegal_test = open(test_legalfile, 'w+', encoding='UTF-8')

    for batch_idx, (handpai, incre, who, act, actlegal) in enumerate(testloader):
        hand = list(map(int, handpai))
        incre = list(map(int, incre))
        who = list(map(int, who))
        all = who + incre
        act = int(act)
        res = []
        res = '[' + str(hand) + ',' + str(all) + ',' + str(act) + ']' + '\n'
        print(res)
        reslegal = list(map(int, actlegal))
        num1 = reslegal.count(1)
        if num1 == 1:
            continue
        reslegal = [i for i, x in enumerate(reslegal) if x == 1]
        while len(reslegal) < max:
            reslegal.append(reslegal[0])
        reslegal = str(reslegal) + '\n'
        f_test.write(res)
        flegal_test.write(reslegal)
    f_test.close()
    flegal_test.close()

dir1 = './p0winresult_tezheng'
dir2 = './p0winresult'
datatxt='result.txt'
legaltxt='legal.txt'
time_start = time.time()
#write_delete(dir1,dir2,datatxt,legaltxt)
#writefile(dir1,dir2,datatxt,legaltxt)
trainfile='train.txt'
legalfile='trainlegal.txt'
testfile='test.txt'
test_legalfile='testlegal.txt'
#dividedataset(datatxt,legaltxt,trainfile,legalfile,testfile,test_legalfile)
time_end = time.time()
print('time cost:', time_end - time_start)










