const express=require('express')
const router=express.Router();
router.get('/router',(req,res)=>{
    res.send('router')
})
module.exports=router;