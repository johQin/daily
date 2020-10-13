/**
 * 题目：输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
 * 思路：
 * 1.首先判定链表为空的情况，如果pHead1为空，返回pHead2，反之亦然。
 * 2.两个链表不为空，比较pHead看谁val小，小的作为起点，把起点记住为head，
 * 然后再令一个small的指针指向已经合并链表的尾部，初始时small为head。然后作为起点的一个链表的头向后移一个，开始合并工作
 * 3.pHead的val小的，samll.next=pHead，然后small后移到pHead，pHead后移一个。重复。
 * 4.当pHead任意一个为null时，说明有一个链表已经走完了，把small.next令为另一个未走完的pHead。
 */
function Merge(pHead1, pHead2)
{
    if(pHead1==null){
        return pHead2;
    }
    if(pHead2==null){
        return pHead1;
    }
    let head=null;
    if(pHead1.val>pHead2.val){
        head=pHead2;
        pHead2=pHead2.next;
    }else{
        head=pHead1;
        pHead1=pHead1.next;
    }
    let small=head;
    while(pHead1!=null&&pHead2!=null){
        if(pHead1.val<pHead2.val){
            small.next=pHead1;
            small=pHead1;
            pHead1=pHead1.next;
        }else{
            small.next=pHead2;
            small=pHead2;
            pHead2=pHead2.next;
        }
    }
    if(pHead1==null){
        small.next=pHead2;
    }
    if(pHead2==null){
        small.next=pHead1;
    }
    return head;
}
/**
 * 发散：合并K个排序链表
 * 思路：将k个排序链表两两排序（分治）
    public ListNode mergeKLists(ListNode[] lists) {

        int n=lists.length;
        if(n==0||lists==null) return null;
        //分治
        while(n>1){
            //奇数个时转为偶数个
            if(n%2==1) lists[n-2]=mergeTwoLists(lists[n-2],lists[n-1]);
            //然后两两排序
            for(int i=0;i<n/2;i++){
                lists[i]=mergeTwoLists(lists[2*i],lists[2*i+1]);
            }
            n/=2;
        }
        return lists[0];
    }
 * 
 */