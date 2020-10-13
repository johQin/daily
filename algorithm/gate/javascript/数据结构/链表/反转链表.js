const {head}=require('./LinkNode.js');
/*
输入一个链表，反转链表后，输出新链表的表头
思路：
1.当链表结点只有一个时，直接返回该结点
2.当链表有多个结点时，采用三个指针left，mid，right(首先要判断链表长度为1，为2的情况，然后再做3及以上的情况)
3.将三个指针分别初始化，left=pHead,mid=pHead.next,right=mid.next,并将首结点的left.next赋值为null
4.当right!=null时，将mid.next指向left，node与right之间的链接就断掉了，此后依次将left、node、right往后移(left=mid,mid=right,right=right.next)
5.当right==null时，跳出循环，left为倒数第二个结点，mid为最后一个结点，此时mid.next=null，所以要将mid.next=left
 */
function ReverseList(pHead)
{
    
    if(pHead==null){
        return null;
    }
    let left=pHead,mid=pHead.next;
    if(mid==null){
        return pHead
    }
    let right=mid.next;
    left.next=null;
    while(right!=null){
        mid.next=left;
        left=mid;
        mid=right;
        right=right.next;
    }
    mid.next=left;
    return mid
}
