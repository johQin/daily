/**
 * 输入一个链表，输出该链表中倒数第k个节点
 * 思路：
 * 1.借助两个指针left和right，初始化两个指针为头结点
 * 2.然后让两个指针，间隔k个距离，如果链表比k短，那么直接返回null
 * 3.然后让间隔k个距离的两个指针，同时向后移动，直到right指针为null，返回left
 */

function FindKthToTail(head, k)
{
    if(k<0||head==null){
        return null;
    }
    let left=head,right=head,count=0;
    while(right!=null&&count<k){
        count++;
        right=right.next
    }
    if(count<k){
        return null;
    }
    while(right!=null){
        left=left.next;
        right=right.next;
    }
    return left;
}