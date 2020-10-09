'''
递推公式
青蛙跳台阶
题目：一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个n级的台阶总共有多少种跳法？(先后词序不同也算不同的结果)
答题思路
如果只有1级台阶，那显然只有一种跳法
如果有2级台阶，那么就有2种跳法，一种是分2次跳。每次跳1级，另一种就是一次跳2级
如果台阶级数大于2，设为n的话，这时我们把n级台阶时的跳法看成n的函数，记为,第一次跳的时候有2种不同的选择：
一是第一次跳一级，此时跳法的数目等于后面剩下的n-1级台阶的跳法数目，即为f(n-1)
二是第一次跳二级，此时跳法的数目等于后面剩下的n-2级台阶的跳法数目，即为f(n-2),
因此n级台阶的不同跳法的总数为f(n-1)+f(n-2)，不难看出就是斐波那契数列

变式：若把条件修改成一次可以跳一级，也可以跳2级...也可以跳上n级呢？
f(n)=f(n-1)+f(n-2)+f(n-3)+....+f(2)+f(1)
f(n-1)=f(n-2)+f(n-3)+....+f(2)+f(1)
两式相减得：
f(n)=2f(n-1)
'''
class Frog():
    def jump2Plans(self,n):
        if n==1:
            return 1
        elif n==2:
            return 2
        elif n>2:
            a=1
            b=2
            for i in range(n-1):
                c=a+b
                a=b
                b=c
            return c
        else:
            return 0
    def jumpNPlans(self,n):
        if n==1:
            return 1
        elif n>1:
            a=1
            for i in range(n-1):
                b=2*a
                a=b
            return b
        else:
            return 0
        
if __name__=='__main__':
    f=Frog()
    n=9;
    res=f.jump2Plans(n)
    res1=f.jumpNPlans(n)
    print("青蛙一次可以跳一级也可以跳两级，一共有%d级台阶，共有%d种跳跃方案" % (n,res))
    print("青蛙一次可以跳1,2,..n级，一共有%d级台阶，共有%d种跳跃方案" % (n,res1))