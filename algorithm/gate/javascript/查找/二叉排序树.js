//创建二叉排序树
function createBinarySearchTree(arr){
    let len=arr.length;
    let root=null;
    for(let i=0;i<len;i++){
        root=achieve(root,arr[i]);
    }
    return root;
}
//创建时的树的迭代生成函数
function achieve(root,item){
    if(root==null){
        root=new treeNode(item)
    }else if(item<root.val){
        root.left=achieve(root.left,item)
    }else if(item>root.val){
        root.right=achieve(root.right,item)
    }
    return root;
}
//树的中序遍历函数。
function midOrder(root){
    if(root==null){
        return;
    }
    midOrder(root.left);
    console.log(root.val);
    midOrder(root.right);
}
class treeNode{
    constructor(val){
        this.val=val;
        this.left=null;
        this.right=null;
    }
}
function find(key,tree){
    if(tree==null){
        return -1;
    }
    if(key==tree.val){
        return key;
    }else if(key<tree.val){
        return find(key,tree.left)
    }else if(key>tree.val){
        return find(key,tree.right)
    }
}

let arr=[8,5,9,7,6,4,3,12,15,1];
let tree=createBinarySearchTree(arr);
// console.log(tree);
// midOrder(tree);
console.time();
console.log(find(10,tree));
console.timeEnd();


