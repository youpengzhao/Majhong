#麻将action,random选择
import random
from legal_action_judge import legal_action_judge1,legal_action_judge2
import fileinput
from try1 import writefile
from MahjongGB import MahjongFanCalculator


'''def myjudge(s):
    act_legal=[]
    global actions
    global lastcard
    global myturn
    global lastplayer_out
    global outstr
    global act_vect
    ls=['W1','W2','W3','W4','W5','W6','W7','W8','W9','T1','T2','T3','T4','T5','T6',\
        'T7','T8','T9','B1','B2','B3','B4','B5','B6','B7','B8','B9','F1','F2','F3',\
        'F4','J1','J2','J3']

    fo=open(s,"rt")
    bag=fo.readline()
    while bag:
        myturn = 0  # 是否是我的回合
        lastplayer_out = 1  # 最后打出的牌是否是上家出的
        lastcard = 26  # 最后一张打出的牌/自己摸到的牌
        outstr = ""
        if bag is not '\n':
            bag = bag.rstrip('\n')  # 移除行尾换行符
            actions = []  # 合法指令集
            act_vect = [0] * 235  # 合法指令集向量形式
            hand = []
            show = []
            hand_str = []
            show_str = []
            hhh = []
            print(eval(bag))
            for ca in eval(bag)[0][0]:
                hand.append(ls.index(ca) + 1)
            if eval(bag)[0][1]:
                for show_bag in eval(bag)[0][1]:
                    for ch in show_bag:
                        hh = ls.index(ch) + 1
                        hhh.append(hh)
                    show.append(tuple(hhh[:]))
                    del hhh[:]
            if eval(bag)[0][9][0] == 0:
                myturn = 1
                lastplayer_out = 0
                lastcard = hand[-1]
                del hand[-1]
            elif eval(bag)[0][9][0] == 3:
                myturn = 0
                lastplayer_out = 1
                lastcard = ls.index(eval(bag)[0][8][-1]) + 1
            elif eval(bag)[0][9][0] == 1:
                myturn = 0
                lastplayer_out = 0
                lastcard = ls.index(eval(bag)[0][6][-1]) + 1
            elif eval(bag)[0][9][0] == 2:
                myturn = 0
                lastplayer_out = 0
                lastcard = ls.index(eval(bag)[0][7][-1]) + 1

            for ch in hand:
                hand_str.append(ls[ch - 1])
            if show:
                for ch in show:
                    if len(ch) == 4:
                        show_str.append(("GANG", ls[ch[0] - 1], 1))
                    elif ch[0] == ch[1]:
                        show_str.append(("PENG", ls[ch[1] - 1], 1))
                    else:
                        show_str.append(("CHI", ls[ch[1] - 1], 1))

            if myturn:
                mycard = [0] * 34
                for i in range(len(hand)):  # 手牌初始化:hand中的牌存入mycard
                    mycard[hand[i] - 1] += 1
                mycard[lastcard - 1] += 1

                for i in range(34):  # 暗杠判定
                    if mycard[i] == 4:
                        actions.append(i + 70)

                if show:
                    for i in range(len(show)):  # 补杠判定
                        if len(show[i]) == 3:
                            if show[i][0] == lastcard and show[i][1] == lastcard:
                                actions.append(lastcard + 103)
                for i in range(34):  # 出牌
                    if mycard[i] > 0:
                        actions.append(i + 1)
                try:  # 胡牌
                    counter_fan = 0
                    ans = MahjongFanCalculator(tuple(show_str), tuple(hand_str), ls[lastcard - 1], 0, myturn, False,
                                               False,
                                               True, 0, 0)
                    for i in range(len(ans)):
                        counter_fan = counter_fan + ans[i][0]
                    if counter_fan >= 8:
                        actions.append(35)
                    else:
                        print("胡，但小于8番")
                except Exception as err:
                    print(err)
                else:
                    print(ans)
            else:
                mycard = [0] * 34
                for i in range(len(hand)):  # 手牌初始化
                    if hand[i] > 0:
                        mycard[hand[i] - 1] += 1

                actions.append(0)  # Pass
                act_vect[0] = 1

                try:  # 胡牌
                    counter_fan = 0
                    ans = MahjongFanCalculator(tuple(show_str), tuple(hand_str), ls[lastcard - 1], 0, myturn, False,
                                               False,
                                               True, 0, 0)
                    for i in range(len(ans)):
                        counter_fan = counter_fan + ans[i][0]
                    if counter_fan >= 8:
                        actions.append(35)
                    else:
                        print("胡，但小于8番")
                except Exception as err:
                    print(err)
                else:
                    # print(ans)
                    print('总番数:{}'.format(counter_fan))

                if mycard[lastcard - 1] == 3:  # 明杠判断
                    actions.append(lastcard + 35)

                if mycard[lastcard - 1] >= 2:  # 碰判断
                    mycard[lastcard - 1] -= 2
                    for i in range(34):
                        if mycard[i] > 0:
                            actions.append(i + 138)
                    mycard[lastcard - 1] += 2

                if lastplayer_out:
                    if 1 <= lastcard <= 7:  # 吃（万，左）
                        if mycard[lastcard] > 0 and mycard[lastcard + 1] > 0:
                            actions.append(lastcard + 171)
                            #mycard[lastcard] -= 1
                            #mycard[lastcard + 1] -= 1
                    if 2 <= lastcard <= 8:  # 吃（万，中）
                        if mycard[lastcard - 2] > 0 and mycard[lastcard] > 0:
                            actions.append(lastcard + 177)
                            #mycard[lastcard - 2] -= 1
                            #mycard[lastcard] -= 1
                    if 3 <= lastcard <= 9:  # 吃（万，右）
                        if mycard[lastcard - 3] > 0 and mycard[lastcard - 2] > 0:
                            actions.append(lastcard + 183)
                            #mycard[lastcard - 2] -= 1
                            #mycard[lastcard - 3] -= 1

                    if 10 <= lastcard <= 16:  # 吃（条，左）
                        if mycard[lastcard] > 0 and mycard[lastcard + 1] > 0:
                            actions.append(lastcard + 183)
                            #mycard[lastcard] -= 1
                            #mycard[lastcard + 1] -= 1
                    if 11 <= lastcard <= 17:  # 吃（条，中）
                        if mycard[lastcard - 2] > 0 and mycard[lastcard] > 0:
                            actions.append(lastcard + 189)
                            #mycard[lastcard - 2] -= 1
                            #mycard[lastcard] -= 1
                    if 12 <= lastcard <= 18:  # 吃（条，右）
                        if mycard[lastcard - 3] > 0 and mycard[lastcard - 2] > 0:
                            actions.append(lastcard + 195)

                    if 19 <= lastcard <= 25:  # 吃（饼，左）
                        if mycard[lastcard] > 0 and mycard[lastcard + 1] > 0:
                            actions.append(lastcard + 195)
                            #mycard[lastcard] -= 1
                            #mycard[lastcard + 1] -= 1
                    if 20 <= lastcard <= 26:  # 吃（饼，中）
                        if mycard[lastcard - 2] > 0 and mycard[lastcard] > 0:
                            actions.append(lastcard + 201)
                            #mycard[lastcard - 2] -= 1
                            #mycard[lastcard] -= 1
                    if 21 <= lastcard <= 27:  # 吃（饼，右）
                        if mycard[lastcard - 3] > 0 and mycard[lastcard - 2] > 0:
                            actions.append(lastcard + 207)
                            #mycard[lastcard - 2] -= 1
                            #mycard[lastcard - 3] -= 1

            if actions:
                print(actions)
                for i in actions:
                    act_vect[i] = 1
                # print("action vector:", act_vect)
            else:
                print("actions is empty")
            
            random.seed()
            outnum = random.choice(actions)
            

            if outnum == 0:
                outstr = "PASS"
            elif 1 <= outnum <= 9:
                outstr = "PLAY W" + str(outnum)
            elif 10 <= outnum <= 18:
                outstr = "PLAY T" + str(outnum - 9)
            elif 19 <= outnum <= 27:
                outstr = "PLAY B" + str(outnum - 18)
            elif 28 <= outnum <= 31:
                outstr = "PLAY F" + str(outnum - 27)
            elif 32 <= outnum <= 34:
                outstr = "PLAY J" + str(outnum - 31)
            elif outnum == 35:
                outstr = "HU"
            elif 36 <= outnum <= 69:  # 明杠
                outstr = "GANG"
            elif 70 <= outnum <= 78:  # 暗杠
                outstr = "GANG W" + str(outnum - 69)
            elif 79 <= outnum <= 87:
                outstr = "GANG T" + str(outnum - 78)
            elif 88 <= outnum <= 96:
                outstr = "GANG B" + str(outnum - 87)
            elif 97 <= outnum <= 100:
                outstr = "GANG F" + str(outnum - 96)
            elif 101 <= outnum <= 103:
                outstr = "GANG J" + str(outnum - 100)
            elif 104 <= outnum <= 112:  # 补杠
                outstr = "BUGANG W" + str(outnum - 103)
            elif 113 <= outnum <= 121:
                outstr = "BUGANG T" + str(outnum - 112)
            elif 122 <= outnum <= 130:
                outstr = "BUGANG B" + str(outnum - 121)
            elif 131 <= outnum <= 134:
                outstr = "BUGANG F" + str(outnum - 130)
            elif 135 <= outnum <= 137:
                outstr = "PENG J" + str(outnum - 134)
            elif 138 <= outnum <= 146:  # 碰
                outstr = "PENG W" + str(outnum - 137)
            elif 147 <= outnum <= 155:
                outstr = "PENG T" + str(outnum - 146)
            elif 156 <= outnum <= 164:
                outstr = "PENG B" + str(outnum - 155)
            elif 165 <= outnum <= 168:
                outstr = "PENG F" + str(outnum - 164)
            elif 169 <= outnum <= 171:
                outstr = "PENG J" + str(outnum - 168)
            elif outnum == 172 or outnum == 179 or outnum == 186:  # 吃（万）
                outstr = "CHI W2 "
            elif outnum == 173 or outnum == 180 or outnum == 187:
                outstr = "CHI W3 "
            elif outnum == 174 or outnum == 181 or outnum == 188:
                outstr = "CHI W4 "
            elif outnum == 175 or outnum == 182 or outnum == 189:
                outstr = "CHI W5 "
            elif outnum == 176 or outnum == 183 or outnum == 190:
                outstr = "CHI W6 "
            elif outnum == 177 or outnum == 184 or outnum == 191:
                outstr = "CHI W7 "
            elif outnum == 178 or outnum == 185 or outnum == 192:
                outstr = "CHI W8 "
            elif outnum == 193 or outnum == 200 or outnum == 207:  # 吃（条）
                outstr = "CHI T2 "
            elif outnum == 194 or outnum == 201 or outnum == 208:
                outstr = "CHI T3 "
            elif outnum == 195 or outnum == 202 or outnum == 209:
                outstr = "CHI T4 "
            elif outnum == 196 or outnum == 203 or outnum == 210:
                outstr = "CHI T5 "
            elif outnum == 197 or outnum == 204 or outnum == 211:
                outstr = "CHI T6 "
            elif outnum == 198 or outnum == 205 or outnum == 212:
                outstr = "CHI T7 "
            elif outnum == 199 or outnum == 206 or outnum == 213:
                outstr = "CHI T8 "
            elif outnum == 214 or outnum == 221 or outnum == 228:  # 吃（饼）
                outstr = "CHI B2 "
            elif outnum == 215 or outnum == 222 or outnum == 229:
                outstr = "CHI B3 "
            elif outnum == 216 or outnum == 223 or outnum == 230:
                outstr = "CHI B4 "
            elif outnum == 217 or outnum == 224 or outnum == 231:
                outstr = "CHI B5 "
            elif outnum == 218 or outnum == 225 or outnum == 232:
                outstr = "CHI B6 "
            elif outnum == 219 or outnum == 226 or outnum == 233:
                outstr = "CHI B7 "
            elif outnum == 220 or outnum == 227 or outnum == 234:
                outstr = "CHI B8 "

            # print("out string action is : {}".format(outstr))
            if outnum >= 172:
                i = random.randint(0, 33)
                while mycard[i] == 0:
                    i = i + 1
                    if i == 34:
                        i = 0
                if 0 <= i <= 8:
                    outstr = outstr + "W" + str(i + 1)
                elif 9 <= i <= 17:
                    outstr = outstr + "T" + str(i - 8)
                elif 18 <= i <= 26:
                    outstr = outstr + "B" + str(i - 17)
                elif 27 <= i <= 30:
                    outstr = outstr + "F" + str(i - 26)
                elif 31 <= i <= 33:
                    outstr = outstr + "J" + str(i - 30)
            print("out string action is : {}".format(outstr))
            act_legal.append(act_vect)

        bag = fo.readline()
    fo.close()
    return act_legal'''


def myjudge(s):
    legal=[]
    for bag in fileinput.input(s):
        if bag is not '\n':
            print(bag)
            bag = bag.rstrip('\n')  # 移除行尾换行符
            if eval(bag)[0][9][0]==4:
                bag=eval(bag)
                act_legal=legal_action_judge2(bag)
                legal.append(act_legal)
                #act_legal =legaltransform(act_legal)
            else:
                bag=eval(bag)
                act_legal=legal_action_judge1(bag)
                legal.append(act_legal)
                #print(act_legal)
                #act_legal = legaltransform(act_legal)
    return legal


'''if __name__ == "__main__":
    dir='./p0winresult'
    domin='legal.txt'
    writefile(dir,domin)
    a=myjudge('legal.txt')
    print(a[0])'''


