'''
0~n区间内1出现的次数
eg：0~13内有1,10,11,12,13有1的数字，1共出现6次
思路：
对数字n按位分析，对每位出现1的次数进行累加，最后得到1出现的次数。
eg：
432045134——high-cur-low，e为此位的幂次，个位e=0，十位e=1....
前提：要让此位为出现1，并且数字小于n
1.cur=0时，必须向高位借1，让这位满足前提条件，1出现的次数为
高位排列(high+1-1)，+1意味着高位有0~high的数，-1是因为cur位借了1
低位排列(10^e)，低位共有10^e排列
2.cur=1时，
3.cur>1时，
'''
def achieve(n):
    preceive=1
    low=1
    cur=1
    high=1
    e=0
    res=0
    while high!=0:
        high=n//(preceive*10)#“//”为整除
        cur=(n//preceive)%10
        low=n%preceive
        preceive*=10

        if cur==0:
            num=(high+1-1)*pow(10,e)
        elif cur>1:
            num=(high+1)*pow(10,e)
        else:
            num=high*pow(10,e)+low+1
        
        res+=num
        e+=1

    return res
if __name__=="__main__":
    print(achieve(13))