const {head}=require('./LinkNode.js');

function achieve(head){
    if(head==null){
        return -1;
    }
    let h=head,arr=[],count=0,res=[];
    //入栈
    while(h){
        arr.push(h.val);
        h=h.next;
        count++;
    }
    //出栈
    while(count>0){
        res.push(arr.pop());
        count--;
    }
}

achieve(head);


