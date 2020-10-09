def ReplaceAdd(m,n):
    while m!=0:
        tmp=m
        m=(m&n)<<1#进位
        n=(tmp^n)&0xffffffff #异或
    return n if n<=0x7fffffff else ~(n^0xffffffff)

if __name__=='__main__':
    m=-200
    n=-400
    print(ReplaceAdd(m,n)) 