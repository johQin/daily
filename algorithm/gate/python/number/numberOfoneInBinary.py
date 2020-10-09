'''
题目：输入一个整数，输出该数二进制表示中1的个数。其中负数用补码显示
补码：整数不变，负数是其正数的反码+1
'''
#法一
def numberOfoneM1(n):
    n=0xFFFFFFFF & n#取n的后32位
    count=0
    print('n的二进制：%s' % bin(n))
    for i in str(bin(n)):
        if i=='1':
            count+=1
    return count
#法二
def numberOfoneM2(n):
    n=0xFFFFFFFF & n
    count=0
    for i in range(32):
        mask=1 << i
        if n & mask:
            count+=1
    return count
def numberOfoneM3(n):
    count=0
    while n:
        n=n&(n-1)#n & n-1与一次就会少个1
        count+=1
        n=0xFFFFFFFF & n #考虑负数情况
    return count
if __name__ in '__main__':
    n=-2
    print('字符串方法方法：整数：%d，它的二进制有%d个1' % (n,numberOfoneM1(n)) )
    print('&1方法方法：整数：%d，它的二进制有%d个1' % (n,numberOfoneM2(n)) )
    print('&(n-1)方法：整数：%d，它的二进制有%d个1' % (n,numberOfoneM3(n)) )
