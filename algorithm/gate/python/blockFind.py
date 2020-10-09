
'''
分块查找：
分块查找又称索引顺序查找，分块建立索引，块内顺序查找。块内无序，块间有序。
通过建立索引表，即记录每个块中的最大值和该块的起始元素索引值。
例如：[24,21,6,11,8,22,|32,31,54,|72,61,78,|88,83]
索引表：{24:0,54:6,78:9,88:12},key为块内最大值，value为块的起始索引
先通过索引表确定查找的范围，确定范围后，再在块内进行顺序查找。
'''
class BlockFind():
    def achieve(self,dic,arr,target):
        Bdic=dic.items()#[(KEY1, VAL1), (KEY2, VAL2), (KEY3, VAL3)]
        bmax=dic.keys()
        bindex=dic.values()
        for i in 


if __name__=='__main__':
