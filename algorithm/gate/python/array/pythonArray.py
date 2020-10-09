if __name__=='__main__':
    tlist=[1,2,3,4]
    #都是对原数组的操作
    tlist.append(6)
    tlist.extend([7,8,9])
    tlist.insert(4,5)
    print("数组添加操作，append(obj),extend(seq),insert(index,obj),当前数组：%s" % tlist)
    tlist.pop()#等价于tlist.pop(-1)，python中pop(index)，默认index=-1，弹出list的最后一个元素
    tlist.pop(0)#弹出list的第一个元素，少了1
    tlist.pop(1)#弹出索引为1的元素，少了3
    tlist.remove(8)#remove(obj),删除第一个匹配obj的元素
    print("数组删除操作，pop(index),当前数组：%s" % tlist)

    tlist.reverse()
    print("数组反向：%s" % tlist)

    print('匹配对象的索引：',tlist.index(6))#找到第一个匹配项的索引，如果没有找到对象则抛出异常。
    print('匹配对象的出现的次数：',tlist.count(6))