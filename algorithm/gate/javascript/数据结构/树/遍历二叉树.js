const {TreeRoot}=require('./TreeNode.js');
function preOrderRecurive(root){
    if(root==null){
        return null;
    }
    console.log(root.val);
    preOrderRecurive(root.left);
    preOrderRecurive(root.right);
}
function midOrderRecurive(root){
    if(root==null){
        return null
    }
    midOrderRecurive(root.left);
    console.log(root.val);
    midOrderRecurive(root.right);
}
function latOrderRecurive(root){
    if(root==null){
        return null
    }
    latOrderRecurive(root.left);
    latOrderRecurive(root.right);
    console.log(root.val);
}
function preOrder(root){
    if(root==null){
        return null
    }
    let tmpNode=root,stack=[],node=null;
    while(tmpNode||stack.length>0){
        while(tmpNode){
            console.log(tmpNode.val);
            stack.push(tmpNode);
            tmpNode=tmpNode.left;
        }
        node=stack.pop();
        tmpNode=node.right;
    }
}