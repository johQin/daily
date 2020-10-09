'''
斐波那契数列
递推公式
如果在使用递归的时候涉及到重复计算，
如果“前面'计算'出的结果”对后面的计算有影响，那么我们通常会保留此结果，以供给下一次计算使用。
增加临时变量，存储结果
'''
class Resolve():
    def fibonacci(self,n):
        if n==0 :
            return 0
        elif n==1 :
            return 1
        elif n>1:
            a=0
            b=1
            c=0
            for i in range(0,n-1):
                c=a+b
                a=b
                b=c
            return c
        else:
            return "param is invalid"
    def complexFibonacci(self,n):
        if n==0:
            return 0
        elif n==1:
            return 1
        elif n>1:
            return self.complexFibonacci(n-1)+self.complexFibonacci(n-2)
        else:
            return "param is invalid"
if __name__ =='__main__' :
    fb=Resolve()
    n=4
    res=fb.fibonacci(n)
    print("正确的算法之斐波那契数列的第%d个数是%d,算法复杂度为O(n)"% (n,res)) 
    res2=fb.complexFibonacci(n)
    print("错误的算法之斐波那契数列的第%d个数是%d,算法复杂度为O(2的n次方)"%(n,res2))
"""
python的占位符操作：https://www.cnblogs.com/a389678070/p/9559360.html
常见的占位符有：
%d    整数
%f    浮点数
%s    字符串
%x    十六进制整数
tpl = "i am %s" % "alex"
tpl = "i am %s age %d" % ("alex", 18)
注意：
1.模板字符串和参数之间以 % 号相隔
2.多个参数用括号括起来
3.还有format用法
"""
    