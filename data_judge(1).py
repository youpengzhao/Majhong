#麻将洗出来的数据judge
#可以连续检测多个文件
#PENG card1从碰后打出card1改回碰card1
#CHI 和 PENG 后进入状态4
#读写每个文件分别打开、关闭一次。（处理多个文件内存也不会溢出）
#修复了补杠只能补lastcard的问题
from MahjongGB import MahjongFanCalculator
import time
myturn = 0             #是否是我的回合
lastplayer_out = 1     #最后打出的牌是否是上家出的
lastcard=26             #最后一张打出的牌/自己摸到的牌


def myjudge():
    
    global lastcard
    global myturn
    global lastplayer_out
    ls=['W1','W2','W3','W4','W5','W6','W7','W8','W9','T1','T2','T3','T4','T5','T6',\
        'T7','T8','T9','B1','B2','B3','B4','B5','B6','B7','B8','B9','F1','F2','F3',\
        'F4','J1','J2','J3']
    for txtid in range(100,364):#文件序号范围
        #print("=="*30)
        #print("=="*30)
        ftxt=open("data_judge_result"+str((txtid-1)//10000)+".txt","at")#打开记录结果的文本文件
        txtname="("+str(txtid)+").txt"
        fo=open("G:/code/p0winresult/"+txtname,"rt",encoding='UTF-8')
        line_counter=0#行数计数器
        if txtid%100==0:
            print(txtname)
        #ftxt.write("=="*30+'\n')
        #ftxt.write(txtname+"\n\n")
        #bag=fo.readline()#读一行
        for bag in fo:#读所有行
            actions=[]             #合法指令集
            act_vect=[0]*235       #合法指令集向量形式
            hand=[]
            show=[]
            hand_str=[]
            show_str=[]
            hhh=[]
            line_counter+=1
            
            #print("this is line ",line_counter)
            if not eval(bag)[0][9]:
                ftxt.write("=="*30+'\n')
                ftxt.write("txtname: "+txtname+"  line:"+str(line_counter)+' 当前玩家回合:空\n')
                ftxt.write(bag)#bag里应该自带一个换行符
                continue
            elif eval(bag)[0][9][0]==4:#状态4
                mycard=[0]*34
                for ca in eval(bag)[0][0]:
                    mycard[ls.index(ca)]+=1
                    hand.append(ls.index(ca)+1)
                if eval(bag)[0][1]:#展示的牌
                    for show_bag in eval(bag)[0][1]:
                        for ch in show_bag:
                            hh=ls.index(ch)+1
                            hhh.append(hh)
                        show.append(tuple(hhh[:]))
                        del hhh[:]

                for i in range(34):
                    if mycard[i]>0:
                        actions.append(i+1)            
            else:#不是状态4
                for ca in eval(bag)[0][0]:
                    hand.append(ls.index(ca)+1)
                if eval(bag)[0][1]:#展示的牌
                    for show_bag in eval(bag)[0][1]:
                        for ch in show_bag:
                            hh=ls.index(ch)+1
                            hhh.append(hh)
                        show.append(tuple(hhh[:]))
                        del hhh[:]
                if eval(bag)[0][9][0]==0:
                    myturn=1
                    lastplayer_out=0
                    lastcard=hand[-1]
                    del hand[-1]#手牌数组中删除新摸到的牌
                elif eval(bag)[0][9][0]==3:
                    myturn=0
                    lastplayer_out=1
                    lastcard=ls.index(eval(bag)[0][8][-1])+1
                elif eval(bag)[0][9][0]==1:
                    myturn=0
                    lastplayer_out=0
                    lastcard=ls.index(eval(bag)[0][6][-1])+1
                elif eval(bag)[0][9][0]==2:
                    myturn=0
                    lastplayer_out=0
                    lastcard=ls.index(eval(bag)[0][7][-1])+1
    
                for ch in hand:
                    hand_str.append(ls[ch-1])
                if show:
                    for ch in show:
                        if len(ch)==4:
                            show_str.append(("GANG",ls[ch[0]-1],1))
                        elif ch[0]==ch[1]:
                            show_str.append(("PENG",ls[ch[1]-1],1))
                        else:
                            show_str.append(("CHI",ls[ch[1]-1],1))

                if myturn :#我的回合
                    mycard=[0]*34
                    for i in range(len(hand)):#手牌初始化:hand中的牌存入mycard
                        mycard[hand[i]-1] +=1
                    mycard[lastcard-1] +=1
                    for i in range(34):#暗杠判定
                        if mycard[i]==4 :
                            actions.append(i+70)

                    if show:
                        for i in range(len(show)):#补杠判定
                            if len(show[i])==3:
                                if show[i][0]==show[i][1] and show[i][1] in hand:
                                    actions.append(show[i][0]+103)
                                elif show[i][0]==show[i][1] and show[i][1] == lastcard:
                                    actions.append(show[i][0]+103)
                    for i in range(34):#出牌
                        if mycard[i]>0:
                            actions.append(i+1)
                    try:#胡牌
                        counter_fan=0
                        ans=MahjongFanCalculator(tuple(show_str),tuple(hand_str),ls[lastcard-1],0,True,False,False,False,0,0)
                        for i in range(len(ans)):
                            counter_fan=counter_fan+ans[i][0]
                        if counter_fan>=8:
                            actions.append(35)
                        else:#胡牌小于8番
                            #print("胡，但小于8番")
                            if eval(bag)[1]==35:
                                #ftxt.write("=="*30+'\n')
                                #ftxt.write("txtname: "+txtname+"  line:"+str(line_counter)+' 番数为'+str(counter_fan)+',小于8\n')
                                #ftxt.write(bag)#bag里应该自带一个换行符
                                fdeln=open("should_delete.txt","at")
                                fdeln.write(str(txtid)+'\n')
                                fdeln.close()
                                continue
                            pass
                    except Exception as err:
                        #print(err)
                        if str(err)=="ERROR_WRONG_TILES_COUNT":
                            ftxt.write("=="*30+'\n')
                            ftxt.write("txtname: "+txtname+"  line:"+str(line_counter)+' ERROR_WRONG_TILES_COUNT\n')
                            ftxt.write(bag)#bag里应该自带一个换行符
                            continue
                        elif eval(bag)[1]==35 and str(err)=="ERROR_NOT_WIN":
                            ftxt.write("=="*30+'\n')
                            ftxt.write("txtname: "+txtname+"  line:"+str(line_counter)+' ERROR_NOT_WIN\n')
                            ftxt.write(bag)#bag里应该自带一个换行符
                            continue
                    else:
                        #print(ans)
                        pass       
                else:#不是我的回合
                    mycard=[0]*34
                    for i in range(len(hand)):#手牌初始化
                        if hand[i]>0:
                            mycard[hand[i]-1] +=1

                    actions.append(0)#Pass

                    try:#胡牌
                        counter_fan=0
                        ans=MahjongFanCalculator(tuple(show_str),tuple(hand_str),ls[lastcard-1],0,False,False,False,False,0,0)
                        for i in range(len(ans)):
                            counter_fan=counter_fan+ans[i][0]
                        if counter_fan>=8:
                            actions.append(35)
                        else:
                            #print("胡，但小于8番")
                            if eval(bag)[1]==35:
                                #ftxt.write("=="*30+'\n')
                                #ftxt.write("txtname: "+txtname+"  line:"+str(line_counter)+' 番数为'+str(counter_fan)+',小于8\n')
                                #ftxt.write(bag)#bag里应该自带一个换行符
                                fdeln=open("should_delete.txt","at")
                                fdeln.write(str(txtid)+'\n')
                                fdeln.close()
                                continue
                            pass
                    except Exception as err:
                        #print(err)
                        if str(err)=="ERROR_WRONG_TILES_COUNT":
                            ftxt.write("=="*30+'\n')
                            ftxt.write("txtname: "+txtname+"  line:"+str(line_counter)+' ERROR_WRONG_TILES_COUNT\n')
                            ftxt.write(bag)#bag里应该自带一个换行符
                            continue
                        elif eval(bag)[1]==35 and str(err)=="ERROR_NOT_WIN":
                            ftxt.write("=="*30+'\n')
                            ftxt.write("txtname: "+txtname+"  line:"+str(line_counter)+' ERROR_NOT_WIN\n')
                            ftxt.write(bag)#bag里应该自带一个换行符
                            continue
                    else:
                        #print(ans)
                        #print('总番数:{}'.format(counter_fan))
                        pass

                    if mycard[lastcard-1]==3:#明杠判断
                        actions.append(lastcard+35)

                    if mycard[lastcard-1]>=2:#碰判断
                        actions.append(lastcard+137)

                    if lastplayer_out:
                        if 1<=lastcard<=7:#吃（万，左）
                            if mycard[lastcard]>0 and mycard[lastcard+1]>0:
                                actions.append(lastcard+171)
                                #mycard[lastcard]-=1
                                #mycard[lastcard+1]-=1
                        if 2<=lastcard<=8:#吃（万，中）
                            if mycard[lastcard-2]>0 and mycard[lastcard]>0:
                                actions.append(lastcard+177)
                                #mycard[lastcard-2]-=1
                                #mycard[lastcard]-=1
                        if 3<=lastcard<=9:#吃（万，右）
                            if mycard[lastcard-3]>0 and mycard[lastcard-2]>0:
                                actions.append(lastcard+183)
                                #mycard[lastcard-2]-=1
                                #mycard[lastcard-3]-=1

                        if 10<=lastcard<=16:#吃（条，左）
                            if mycard[lastcard]>0 and mycard[lastcard+1]>0:
                                actions.append(lastcard+183)
                                #mycard[lastcard]-=1
                                #mycard[lastcard+1]-=1
                        if 11<=lastcard<=17:#吃（条，中）
                            if mycard[lastcard-2]>0 and mycard[lastcard]>0:
                                actions.append(lastcard+189)
                                #mycard[lastcard-2]-=1
                                #mycard[lastcard]-=1
                        if 12<=lastcard<=18:#吃（条，右）
                            if mycard[lastcard-3]>0 and mycard[lastcard-2]>0:
                                actions.append(lastcard+195)
                                #mycard[lastcard-2]-=1
                                #mycard[lastcard-3]-=1

                        if 19<=lastcard<=25:#吃（饼，左）
                            if mycard[lastcard]>0 and mycard[lastcard+1]>0:
                                actions.append(lastcard+195)
                                #mycard[lastcard]-=1
                                #mycard[lastcard+1]-=1
                        if 20<=lastcard<=26:#吃（饼，中）
                            if mycard[lastcard-2]>0 and mycard[lastcard]>0:
                                actions.append(lastcard+201)
                                #mycard[lastcard-2]-=1
                                #mycard[lastcard]-=1
                        if 21<=lastcard<=27:#吃（饼，右）
                            if mycard[lastcard-3]>0 and mycard[lastcard-2]>0:
                                actions.append(lastcard+207)
                                #mycard[lastcard-2]-=1
                                #mycard[lastcard-3]-=1

            if actions :
                #print("legal actions:",actions)
                #print("practical action:",eval(bag)[1])
                for i in actions:
                    act_vect[i]=1
                #print("action vector:",act_vect)
            else:
                #print("actions is empty")
                pass
            hand.sort()
            #print("hand:{} ,lastcard:{}".format(hand,lastcard))
            #print("show:",show)
            #print("current player:",eval(bag)[0][9][0])
            #print("lastcard:",lastcard)
            if eval(bag)[1] in actions:
                #print("practical action valid")
                pass
            else:
                #print("practical action invalid")
                #print("\n\n\nplease check now!!\n\n")
                ftxt.write("=="*30+'\n')
                ftxt.write("txtname: "+txtname+"  line:"+str(line_counter)+' practical action invalid\n')
                ftxt.write(bag)#bag里应该自带一个换行符
            #print("--"*30)
    
        fo.close()
        ftxt.close()

if __name__ == "__main__":
    start_time=time.time()
    myjudge()
    end_time=time.time()
    print("运行时间：{:.2f}s".format(end_time-start_time))
