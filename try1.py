import os
import os.path
import fileinput


def writefile(s1,s2,domin1,domin2):
    filedir1 = s1
    filedir2 = s2
    # 获取当前文件夹中的文件名称列表
    filenames = os.listdir(filedir1)
    # 打开当前目录下的result.txt文件，如果没有则创建
    f = open(domin1, 'w')
    #先遍历文件名
    for filename in filenames:
        filepath = filedir1 + '/' + filename
        #print(filepath)
        # 遍历单个文件，读取行数
        for line in open(filepath):
            f.writelines(line)
        f.write('\n')
    # 关闭文件
    f.close()
    f2 = open(domin2, 'w')
    # 先遍历文件名
    for filename in filenames:
        filepath = filedir2 + '/' + filename
        #print(filepath)
        # 遍历单个文件，读取行数
        for line in open(filepath):
            f2.writelines(line)
        f2.write('\n')
    # 关闭文件
    f2.close()


def getdata(s):
    act = []
    handpai = []
    increment = []
    who = []
    catagory=('handpai','increment','who','act')
    for line in fileinput.input(s):
        if line is not '\n':
            rs = line.rstrip('\n')  # 移除行尾换行符
            numbers1 = list(eval(rs)[0])
            numbers1 = list(map(int, numbers1))
            numbers2 = eval(rs)[1]
            numbers2 = list(map(int, numbers2))
            if len(numbers2) == 1:
                numbers2.append(numbers1[-1])
            act.append(eval(rs)[2])
            handpai.append(numbers1)
            incre = [numbers2[1]]
            w = [numbers2[0]]
            increment.append(incre)
            who.append(w)

    dic=dict.fromkeys(catagory)
    dic['handpai']=handpai
    dic['increment']=increment
    dic['who']=who
    dic['act']=act

    return dic


def get_legal(s):
    actlegal = []
    for line in fileinput.input(s):
        if line is not '\n':
            rs = line.rstrip('\n')  # 移除行尾换行符
            numbers1 = list(eval(rs))
            actlegal.append(numbers1)
    return actlegal


def get_state(s):
    state=[]
    for line in fileinput.input(s):
        if line is not '\n':
            rs = line.rstrip('\n')  # 移除行尾换行符
            numbers1 = list(eval(rs))
            state.append(numbers1)
    return state


if __name__ == "__main__":
    '''a=getdata('result.txt')
    print(a)'''
'''dir='./p0winresult_tezheng'
domin='result.txt'
writefile(dir,domin)
name='result.txt'
dic=getdata(name)
print(dic)'''

#print(a[0])

