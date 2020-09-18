import pickle,fileinput,os
import torch
from try1 import getdata,get_legal
from torch.utils import data


def changetezheng(s_txt,s_pkl):
    filetxt=s_txt
    filepkl = s_pkl
    f_in = open(filepkl,'wb')
    dic=getdata(filetxt)
    pickle.dump(dic,f_in)
    f_in.close()


def changelegal(s_txt,s_pkl):
    filetxt = s_txt
    filepkl = s_pkl
    f_in = open(filepkl, 'wb')
    dic = get_legal(filetxt)
    pickle.dump(dic, f_in)
    f_in.close()


def changedir(dir_src,dir_dst,s,s_legal):
    num = int(len(os.listdir(dir_src)) / 2)
    for i in range(1,num+1):
        txt_tezheng = dir_src+s+str(i)+'.txt'
        pkl_tezheng = dir_dst+s+str(i)+'.pkl'
        changetezheng(txt_tezheng,pkl_tezheng)
        txt_legal = dir_src+s_legal+str(i)+'.txt'
        pkl_legal = dir_dst+s_legal+str(i)+'.pkl'
        changelegal(txt_legal,pkl_legal)


class MypklDataset(data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, datapkl, legalpkl):  # 初始化一些需要传入的参数
        super(MypklDataset, self).__init__()
        f_tezheng = open(datapkl, 'rb')
        dic = pickle.load(f_tezheng)
        f_tezheng.close()
        f_legal = open(legalpkl, 'rb')
        actlegal = pickle.load(f_legal)
        f_legal.close()
        self.hand = dic['handpai']
        self.incre = dic['increment']
        self.who = dic['who']
        self.action = dic['act']
        self.legal = actlegal

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

#changefile('test')
'''dir_src = './divide/whole2/'
dir_dst = './dividepkl/whole2/'
s='train'
s_legal='legaltrain'
changedir(dir_src,dir_dst,s,s_legal)'''

'''changelegal('testlegal')
f=open('test.pkl','rb')
b=pickle.load(f)
f.close()
print(b)
f=open('testlegal.pkl','rb')
c=pickle.load(f)
f.close()
print(c)'''
#print(c)
