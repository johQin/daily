'''
丑数：把只包含质因子2，3和5的数
素数（质数）：指在大于1的自然数中，除了1和它本身以外不再有其他因数的自然数
题目描述：
习惯上，我们把1当做第一个丑数，求按从小到大的顺序输出第N个丑数
思路：
数字大的丑数是数字小的丑数乘以2,3或5得来的。而不用通过不断的整除2,3和5得到他是否为丑数
这里我们借助三个指针（分别带有权重2,3,5），并同时指向一个初始数组只有一个丑数1的丑数数组[1]。
然后通过不断乘且向后移动指针，并按从小到大的顺序排列，将会得到不断增长的丑数数组
'''
def uglyArray(n):
    if n==0:
        return 0
    if n==1:
        return 1
    uarr=[1]
    two=0
    three=0
    five=0
    count=1
    while count<n:
        ug=min(uarr[two]*2,uarr[three]*3,uarr[five]*5)
        uarr.append(ug)
        count+=1
        if ug==uarr[two]*2:
            two+=1
        if ug==uarr[three]*3:
            three+=1
        if ug==uarr[five]*5:
            five+=1
    print("当前计算到的丑数数组：{0}".format(uarr))
    print("第%d个丑数为%d" % (n,uarr[n-1]))
    return uarr[n-1]
if __name__ =="__main__":
    uglyArray(10)
