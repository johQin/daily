
/*由遍历序列构造二叉树
原理：由二叉树的先序序列（或后序序列或层序序列）和中序序列可以唯一的确定一颗二叉树
思路：
一、先中
1.先序序列的第一个元素一定是二叉树的根节点，这个根节点将中序序列分成左右两个子序列，
    左序列就是根节点的左子树，
    右序列就是根节点的右子树。
2.根据上面从中序序列分出来的左右序列，在先序序列中找到对应的两个序列左先序列和右先序列，
    左先序列的第一个结点，为左子树的根节点——左子根节点
    右先序列的第一个节点，为右子树的根节点——右子根节点
3.然后根据左子根节点，在左序列中分成两个左-左右子序列
      根据右子根节点，在有序列中分成两个右-左右子序列
4.如此循环分解下去...
二、后中
后序序列的最后一个节点，如同先序序列的第一个节点
三、层中
*/
function TreeNode(x) {
    this.val = x;
    this.left = null;
    this.right = null;
}
function reConstructBinaryTree(pre, vin)
{
    if(pre.length!=vin.length){
        return null
    }
    if(pre.length==0){
        return null
    }
    let rootVal=pre[0];
    let splitVin=vin.indexOf(rootVal);
    let preLeft,preRight,vinLeft,vinRight;
    if(splitVin+1>=pre.length){
        preLeft=pre.slice(1);
        preRight=[];
        vinLeft=vin.slice(0,splitVin);
        vinRight=[];
    }else{
        preLeft=pre.slice(1,splitVin+1);
        preRight=pre.slice(splitVin+1);
        vinLeft=vin.slice(0,splitVin);
        vinRight=vin.slice(splitVin+1);
    }
    
    let root=new TreeNode(rootVal);
    root.left=reConstructBinaryTree(preLeft,vinLeft);
    root.right=reConstructBinaryTree(preRight,vinRight);
    return root;
}