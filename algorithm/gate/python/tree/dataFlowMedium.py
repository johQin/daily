'''
数据流的中位数
中位数是有序列表中间的数。如果列表长度是偶数，中位数则是中间两个数的平均值。
设计一个支持以下两种操作的数据结构：
void addNum(int num) - 从数据流中添加一个整数到数据结构中。
double findMedian() - 返回目前所有元素的中位数。
思路：
1.这种动态（流动）的数据，如果使用数组存储，那么每次新进来一个数据都进行排序的话，效率很低。
处理动态数据来说，一般使用的数据结构是栈、队列、二叉树、堆。在这里还是采用最大最小堆。
2.建立一个类，它有两个属性数组，一个最大堆（用于存储数据流的较小的数），一个最小堆（用于存储数据流较大的数），
3.轮流往最大堆和最小堆放数（保证两个对各占一半的数据，中位数必在堆顶形成），在放之前要与另一个堆比较，
    与最小堆比较的时候，查看要放的数是不是比根节点的值大，如果大，此数放入最小堆，而根结点的值放入最大堆。
    与最大堆比较的时候，查看要放的数是不是比根节点的值小，如果小，此数放入最大堆，而根结点的值放入最小堆。
4.如果输入奇数个数，中位数在最大堆的堆顶。如果输入偶数个数，中位数等于（最大堆堆顶+最小堆堆顶）/2
'''
class DataFlowMedium():
    def __init__(self):
        self.maxHeap=[]
        self.minHeap=[]
        self.count=0
    def addNum(self,num):
        maxHeap=self.maxHeap
        minHeap=self.minHeap
        count=self.count
        tmp=0
        if count%2==1:#往最小堆放数，要与最大堆的堆顶做比较
            if num<maxHeap[0]:
                tmp=maxHeap[0]
                self.adjustMaxHeap(num)
            else:
                tmp=num
            self.addMinHeap(tmp)
        else:#往最大堆添数，要与最小堆堆顶做比较
            if maxHeap==[]:
                tmp=num
            else:
                if minHeap==[]:
                    tmp=num
                else:
                    if minHeap[0]<num:
                        tmp=minHeap[0]
                        self.adjustMinHeap(num)
                    else:
                        tmp=num
            self.addMaxHeap(tmp)
        self.count=count+1
    def getMedium(self):
        if self.count%2==1:
            return self.maxHeap[0]
        else:
            res=(self.maxHeap[0]+self.minHeap[0])/2
            return res
    def addMaxHeap(self,num):#和findKmin中的create相同
        maxHeap=self.maxHeap
        maxHeap.append(num)
        curIndex=len(maxHeap)-1
        while curIndex>0:
            parentIndex=(curIndex-1)>>1
            if maxHeap[parentIndex]<maxHeap[curIndex]:
                maxHeap[curIndex],maxHeap[parentIndex]=maxHeap[parentIndex],maxHeap[curIndex]
                curIndex=parentIndex
            else:
                break
    def addMinHeap(self,num):
        minHeap=self.minHeap
        minHeap.append(num)
        curIndex=len(minHeap)-1
        while curIndex>0:
            parentIndex=(curIndex-1)>>1
            if minHeap[curIndex]<minHeap[parentIndex]:
                minHeap[curIndex],minHeap[parentIndex]=minHeap[parentIndex],minHeap[curIndex]
                curIndex=parentIndex
            else:
                break
    def adjustMaxHeap(self,num):
        maxHeap=self.maxHeap
        length=len(maxHeap)
        if num<maxHeap[0]:
            maxHeap[0]=num
            curIndex=0
            while curIndex<length:
                leftChildIndex=2*curIndex+1
                rightChildIndex=2*curIndex+2
                largerIndex=0
                if rightChildIndex<length:
                    if maxHeap[rightChildIndex]<maxHeap[leftChildIndex]:
                        largerIndex=leftChildIndex
                    else:
                        largerIndex=rightChildIndex
                elif leftChildIndex<length :
                    largerIndex=leftChildIndex
                else:
                    break
                maxHeap[curIndex],maxHeap[largerIndex]=maxHeap[largerIndex],maxHeap[curIndex]
                curIndex=largerIndex
    def adjustMinHeap(self,num):
        minHeap=self.minHeap
        length=len(minHeap)
        if minHeap[0]<num:
            minHeap[0]=num
            curIndex=0
            while curIndex<length:
                leftChildIndex=2*curIndex+1
                rightChildIndex=2*curIndex+2
                smallerIndex=0
                if rightChildIndex<length:
                    if minHeap[rightChildIndex]<minHeap[leftChildIndex]:
                        smallerIndex=rightChildIndex
                    else:
                        smallerIndex=leftChildIndex
                elif leftChildIndex<length:
                    smallerIndex=leftChildIndex
                else:
                    break
                minHeap[curIndex],minHeap[smallerIndex]=minHeap[smallerIndex],minHeap[curIndex]
                curIndex=smallerIndex

if __name__=='__main__':
    dfm=DataFlowMedium()
    for item in [4,10,9,2,15,7,20,19,16,32,45,36]:
        dfm.addNum(item)
        mid=dfm.getMedium()
        # print(dfm.minHeap,dfm.maxHeap)
        print('中位数：%s' % (mid))

