'''
题目描述：
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。
由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
思路：
解法1：通过字典记录每个数出现的次数，然后查看该键它的值是否大于数组长度的一半，如大于则返回，
解法2：
    阵地攻守思想（抵消法）：
    第一个数字作为第一个士兵，守阵地，守阵地的人数c为1，
    当他遇到相同数字的士兵，c++，
    当他遇到不同数字的士兵，他们将互为敌人，一命换一命，c--
    当遇到c=0，阵地无人，新数字又将作为守阵地的人，继续如此。
    到最后还在阵地上的士兵，将“可能”是超过一半的主元素。
    最后，在检查它在数组中，次数是否超过一半
'''
def SuperHalfFreqNum1(arr):#时间复杂度O(n),空间复杂度O(n)
    dictm={}
    halflen=len(arr)>>1
    for i in arr:
        if i in dictm:#判断键是否在字典内
            dictm[i]+=1
        else:
            dictm[i]=1
        if dictm[i]>halflen:
            return i
    return 0
def SuperHalfFreqNum2(arr):#时间复杂度O(n),空间复杂度为O(1)
    last=None
    restCount=0
    halflen=len(arr)>>1
    for i in arr:
        if restCount==0:
            last=i
            restCount=1
        else:
            if last==i:
                restCount+=1
            else:
                restCount-=1
    print('剩余数字',last,restCount)
    if restCount==0:
        return 0
    else:
        restCount=0
        for i in arr:
            if i==last:
                restCount+=1
                if restCount>halflen:
                    return last
        return 0
                

if __name__=='__main__':
    arr=[1,2,3,2,3,6,8,9,2,5,6,2,2,2,2,2,2,2,2]
    print('字典法，得出超过数组长度一半的数字为:%d' % SuperHalfFreqNum1(arr))
    print('抵消法，得出超过数组长度一半的数字为:%d' % SuperHalfFreqNum2(arr))