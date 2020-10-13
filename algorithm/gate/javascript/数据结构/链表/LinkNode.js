class Node{
    constructor(val){
        this.val=val;
        this.next=null;
    }
    toStr(){
        let info=`{val:${this.val},next:${this.next}}`;
        console.log(info)
        return info;
    }
}
let l1=new Node(1);
let l2=new Node(2);
let l3=new Node(3);
let l4=new Node(4);
let l5=new Node(5);
let l6=new Node(6);
let l7=new Node(7);
let l8=new Node(8);
l1.next=l2;
l2.next=l3;
l3.next=l4;
l4.next=l5;
l5.next=l6;
l6.next=l7;
l7.next=l8;
module.exports={
    head:l1,
    Node
}