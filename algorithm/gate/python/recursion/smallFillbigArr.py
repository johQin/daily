'''
用n个2*1的长方形填充2*n长方形，一共有多少种填充法
思路：
1.当竖着放最后一个的时候，还有2*(n-1)的长方形需要填充
2.当横着放最后一个的时候，倒数第二个只能横着放，但还有2*(n-2)的长方形需要填充，
3.故f(n)=f(n-1)+f(n-2)
4.当n=2时，f(2)=2
5.当n=1时，f(1)=1
'''
def smallFillbig(n):
    if n<=0:
        return 0
    elif n==1:
        return 1
    elif n==2:
        return 2
    else:
        a=1
        b=2
        for i in range(2,n):
            c=a+b
            a=b
            b=c
        return c
if __name__=='__main__':
    res=smallFillbig(3)
    print('hah',res)