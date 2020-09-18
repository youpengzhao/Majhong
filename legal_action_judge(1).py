#单个（状态，动作）数据合法指令集生成
from MahjongGB import MahjongFanCalculator

state=[[['T6', 'W7', 'B5', 'T7', 'T5', 'T4', 'T4', 'W7', 'W7', 'B6', 'B4'], [['B4', 'B4', 'B4']], [['B1', 'B2', 'B3']], [['B6', 'B7', 'B8']], [], ['J3', 'F4', 'B1', 'T8'], ['F4', 'B8', 'W1', 'B4', 'B3'], ['F3', 'F2', 'J1', 'J2'], ['F2', 'J1', 'F3', 'F1'], [0]],35]

lastaction=234
def legal_action_judge1(state):
    myturn = 0             #是否是我的回合
    lastplayer_out = 1     #最后打出的牌是否是上家出的
    lastcard=26             #最后一张打出的牌/自己摸到的牌
    ls=['W1','W2','W3','W4','W5','W6','W7','W8','W9','T1','T2','T3','T4','T5','T6',\
        'T7','T8','T9','B1','B2','B3','B4','B5','B6','B7','B8','B9','F1','F2','F3',\
        'F4','J1','J2','J3']
    actions=[]             #合法指令集
    act_vect=[0]*235       #合法指令集向量形式
    hand_num=[]
    show_num=[]
    hand_str=[]
    show_str=[]
    hhh=[]
    bag=state
    for ca in bag[0][0]:
        hand_num.append(ls.index(ca)+1)
    if bag[0][1]:#展示的牌
        for show_bag in bag[0][1]:
            for ch in show_bag:
                hh=ls.index(ch)+1
                hhh.append(hh)
            show_num.append(tuple(hhh[:]))
            del hhh[:]
    if bag[0][9][0]==0:
        myturn=1
        lastplayer_out=0
        lastcard=hand_num[-1]
        del hand_num[-1]
    elif bag[0][9][0]==3:
        myturn=0
        lastplayer_out=1
        lastcard=ls.index(bag[0][8][-1])+1
    elif bag[0][9][0]==1:
        myturn=0
        lastplayer_out=0
        lastcard=ls.index(bag[0][6][-1])+1
    elif bag[0][9][0]==2:
        myturn=0
        lastplayer_out=0
        lastcard=ls.index(bag[0][7][-1])+1

    for ch in hand_num:
        hand_str.append(ls[ch-1])
    if show_num:
        for ch in show_num:
            if len(ch)==4:
                show_str.append(("GANG",ls[ch[0]-1],1))
            elif ch[0]==ch[1]:
                show_str.append(("PENG",ls[ch[1]-1],1))
            else:
                show_str.append(("CHI",ls[ch[1]-1],1))

    if myturn :#我的回合
        mycard=[0]*34
        for i in range(len(hand_num)):#手牌初始化:hand中的牌存入mycard
            mycard[hand_num[i]-1] +=1
        mycard[lastcard-1] +=1
        for i in range(34):#暗杠判定
            if mycard[i]==4 :
                actions.append(i+70)

        if show_num:
            for i in range(len(show_num)):#补杠判定
                if len(show_num[i])==3:
                    if show_num[i][0]==show_num[i][1] and show_num[i][1] in hand_num:
                        actions.append(show_num[i][0]+103)
                    elif show_num[i][0]==show_num[i][1] and show_num[i][1] == lastcard:
                        actions.append(show_num[i][0]+103)
        for i in range(34):#出牌
            if mycard[i]>0:
                actions.append(i+1)
        try:#胡牌
            counter_fan=0
            ans=MahjongFanCalculator(tuple(show_str),tuple(hand_str),ls[lastcard-1],0,myturn,False,False,False,0,0)
            for i in range(len(ans)):
                counter_fan=counter_fan+ans[i][0]
            if counter_fan>=8:
                actions.append(35)
            else:
                #print("胡，但小于8番")
                pass
        except Exception as err:
            #print(err)
            pass
        else:
            #print(ans)
            #print('总番数:{}'.format(counter_fan))
            pass
    else:#不是我的回合
        mycard=[0]*34
        for i in range(len(hand_num)):#手牌初始化
            if hand_num[i]>0:
                mycard[hand_num[i]-1] +=1

        actions.append(0)#Pass
        act_vect[0]=1

        try:#胡牌
            counter_fan=0
            ans=MahjongFanCalculator(tuple(show_str),tuple(hand_str),ls[lastcard-1],0,myturn,False,False,False,0,0)
            for i in range(len(ans)):
                counter_fan=counter_fan+ans[i][0]
            if counter_fan>=8:
                actions.append(35)
            else:
                #print("胡，但小于8番")
                pass
        except Exception as err:
            #print(err)
            pass
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
            if 2<=lastcard<=8:#吃（万，中）
                if mycard[lastcard-2]>0 and mycard[lastcard]>0:
                    actions.append(lastcard+177)
            if 3<=lastcard<=9:#吃（万，右）
                if mycard[lastcard-3]>0 and mycard[lastcard-2]>0:
                    actions.append(lastcard+183)

            if 10<=lastcard<=16:#吃（条，左）
                if mycard[lastcard]>0 and mycard[lastcard+1]>0:
                    actions.append(lastcard+183)
            if 11<=lastcard<=17:#吃（条，中）
                if mycard[lastcard-2]>0 and mycard[lastcard]>0:
                    actions.append(lastcard+189)
            if 12<=lastcard<=18:#吃（条，右）
                if mycard[lastcard-3]>0 and mycard[lastcard-2]>0:
                    actions.append(lastcard+195)

            if 19<=lastcard<=25:#吃（饼，左）
                if mycard[lastcard]>0 and mycard[lastcard+1]>0:
                    actions.append(lastcard+195)
            if 20<=lastcard<=26:#吃（饼，中）
                if mycard[lastcard-2]>0 and mycard[lastcard]>0:
                    actions.append(lastcard+201)
            if 21<=lastcard<=27:#吃（饼，右）
                if mycard[lastcard-3]>0 and mycard[lastcard-2]>0:
                    actions.append(lastcard+207)
    for i in actions:
        act_vect[i]=1
    #print("legal actions:",actions)
    #print(act_vect)
    hand_num.sort()
    #print("hand_num:{} ,lastcard:{}".format(hand_num,lastcard))
    #print("show_num:",show_num)
    #print("current player:",bag[0][9][0])
    return act_vect


def legal_action_judge2(state):
    ls=['W1','W2','W3','W4','W5','W6','W7','W8','W9','T1','T2','T3','T4','T5','T6',\
        'T7','T8','T9','B1','B2','B3','B4','B5','B6','B7','B8','B9','F1','F2','F3',\
        'F4','J1','J2','J3']
    actions=[]             #合法指令集
    act_vect=[0]*235       #合法指令集向量形式
    mycard=[0]*34
    for ch in state[0][0]:
        mycard[ls.index(ch)]+=1


        
    for i in range(34):
        if mycard[i]>0:
            actions.append(i+1)
            act_vect[i+1]=1
    print("legal actions:",actions)
    #print(act_vect)
    return act_vect


if __name__ == "__main__":
    if state[0][9][0]==4:
        legal_action_judge2(state)
    else:
        legal_action_judge1(state)
