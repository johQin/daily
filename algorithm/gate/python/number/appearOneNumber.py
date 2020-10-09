'''
题目描述：一个整型数组里除了两个数字之外，其他的数字都出现了偶数次。请写程序找出这两个出现奇数次的数字
'''
def achieve(arr):
    #将所有的值去异或，因为异或符合结合律，
    #所以出现偶数次的数字，他们相异或都会是0
    #只有那两个单次出现的数字，才会使xorNum不为0
    xorNum=None
    for item in arr:
        if xorNum==None:
            xorNum=item
        else:
            xorNum^=item

    #取出xorNum的二进制数中第一个为1所在位，这个1是两个出现奇数次的数相异或的结果，
    #eg：
    # xorNum二进制为：10100，那么mask就等于100
    # xorNum二进制为：11000，那么mask就等于1000
    # 奇数次出现的数相异或，在mask所在位，一定不相同
    count=0
    while xorNum%2==0:
        xorNum=xorNum>>1
        count+=1
    mask=1<<count

    # 这样我们就可以mask所在位将原数组元素分为两组，一组该位全为1，另一组该位全为0。
    # 再次遍历原数组，mask所在位为0的一起异或，mask所在位为1的一起异或，两组异或的结果分别对应着两个结果。
    firstNum=None
    secondNum=None
    for item in arr:
        if mask&item==0:#一组该位全为0
            if firstNum==None:
                firstNum=item
            else:
                firstNum=firstNum^item
        else:#一组mask位全为1
            if secondNum==None:
                secondNum=item
            else:
                secondNum=secondNum^item
    print("第一个数为：%d，第二个数为：%d" % (firstNum,secondNum))
    return firstNum,secondNum
if __name__ =="__main__":
    achieve([1,3,2,2,2,3,1,5,6,7,6,7,7,7])