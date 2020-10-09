'''
输入n个数，找出其中最小的k个数
思路：
1.如果采用排序做这个，算法复杂度是O(n^2)
2.如果用只有k个元素的数组，并且k个元素按照从小到大的顺序排好（用折半查找法去查询下一个数该放什么位置，复杂度为O( log(n) )）
    在查找到后，还要进行数字的顺序移动复杂度为O(n)，那么整个算法的复杂度为O(nlog(n))
3.树的最大堆和最小堆的方法：
最大堆：一个完全二叉树，根节点的值大于左右子树的关键值。
最小堆：一个完全二叉树，根节点的值小于左右子树的关键值。
这一堆有k个结点，查找算法复杂度为O(log(n)),替换值的算法复杂度为O(log(n))，故整个算法复杂度为O(log(n)*log(n))，

最大堆用于查找最小的k个数，如果一个数比根节点的值小，那么拿出根节点的值，放入这个数进堆，然后再与左右两个子树的比较，放到适合的位置
最小堆用于查找最大的k的数，如果一个数比根节点的值大，那么拿出根节点的值，放入这个数进堆，。。。。

含有n个结点的完全二叉树用数组表示，父结点与左右子结点的索引关系
父结点的索引为i，当2i<=n时，左子结点的索引为2i+1，右子结点的索引为2i+2
子节点的索引为i，父结点的索引为(i-1)整除2 

采用最大堆解决，查找最小的k个数问题
这个过程分两步，
1.创建含有k个结点的初始最大堆，从数组末尾开始创建。
2.与后n-k个值做比较，调整最大堆。


'''
def initCreatMaxHeap(karr):
    maxHeap=[]
    for item in karr:
        maxHeap.append(item)
        curIndex=len(maxHeap)-1
        while curIndex!=0:
            parentIndex=(curIndex-1)>>1
            if maxHeap[parentIndex]<maxHeap[curIndex]:
                maxHeap[curIndex],maxHeap[parentIndex]=maxHeap[parentIndex],maxHeap[curIndex]
                curIndex=parentIndex
            else:
                break
    return maxHeap
def AdjustMaxHeap(maxHeap,num):
    curIndex=0
    length=len(maxHeap)
    if num<maxHeap[0]:
        maxHeap[0]=num
        while curIndex<length:
            leftChildIndex=2*curIndex+1
            rightChildIndex=2*curIndex+2
            largerIndex=0
            if rightChildIndex<length:
                if maxHeap[rightChildIndex]<maxHeap[leftChildIndex]:
                    largerIndex=leftChildIndex
                else:
                    largerIndex=rightChildIndex
            elif leftChildIndex<length:
                largerIndex=leftChildIndex
            else:
                break
            if maxHeap[curIndex]<maxHeap[largerIndex]:
                maxHeap[curIndex],maxHeap[largerIndex]=maxHeap[largerIndex],maxHeap[curIndex]
                curIndex=largerIndex
            else:
                break
def achieve(arr,k):
    if len(arr)<k or k<=0:
        return []
    maxHeap=[]
    i=0
    length=len(arr)
    while i<k:
        maxHeap.append(arr[i])
        i+=1
    maxHeap=initCreatMaxHeap(maxHeap)
    while i<length:
        AdjustMaxHeap(maxHeap,arr[i])
        print('最大堆',maxHeap)
        i+=1

    return maxHeap
if __name__=='__main__':
    arr=[1,7,9,10,25,14,16,98,54,27,5,4,45,36,12]#
    k=10
    achieve(arr,k)

    
