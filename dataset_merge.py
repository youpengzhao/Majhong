import time,os,random
from dataset import MyDataset
from torch.utils.data import ConcatDataset


def mergefile(s1,s2):  #将s1写到s2里
    with open(s1,'r') as f1:
        with open(s2,'a') as f2:
            f2.write(f1.read())
    f1.close()
    f2.close()


def mergedir(test,testlegal,train,trainlegal,dir):
    num = int(len(os.listdir(dir)) / 4)
    idx=[i for i in range(1,num+1)]
    random.shuffle(idx)
    for i in range(num):
        testname=dir+test+str(idx[i])+'.txt'
        testlegalname=dir+testlegal+str(idx[i])+'.txt'
        trainname=dir+train+str(i+1)+'.txt'
        trainlegalname=dir+trainlegal+str(i+1)+'.txt'
        mergefile(testname,trainname)
        mergefile(testlegalname,trainlegalname)

        
'''dir='./divide/whole/'
testfile='test'
testlegal='legaltest'
trainfile='train'
trainlegal='legaltrain'
time_start = time.time()
mergedir(testfile,testlegal,trainfile,trainlegal,dir)
time_end = time.time()
print('time cost:', time_end - time_start)'''
time_start = time.time()
trainfile1='./divide/whole/train1.txt'
trainlegalfile1='./divide/whole/legaltrain1.txt'
train1 =  MyDataset(datatxt=trainfile1, legaltxt=trainlegalfile1)
trainfile2='./divide/whole/train2.txt'
trainlegalfile2='./divide/whole/legaltrain2.txt'
train2 =  MyDataset(datatxt=trainfile2, legaltxt=trainlegalfile2)
trainfile3='traintrans.txt'
trainlegalfile3='legaltrans.txt'
train3 =  MyDataset(datatxt=trainfile3, legaltxt=trainlegalfile3)
print(len(train1),len(train2),len(train3))
a=[]
a.append(train1)
a.append(train2)
a.append(train3)
concat_train = ConcatDataset(a)
#concat_train = ConcatDataset([train1,train2,train3])
print(len(concat_train))
time_end = time.time()
print('time cost:', time_end - time_start)










