const querystring=require('querystring');
function bodyParser(req,res,next){
    let str='';
    req.on('data',(chunk)=>{
        str+=chunk;
    })
    req.on('end',()=>{
        req.body=querystring.parse(str.toString());
        console.log(req.body);
        next()
    })
}
module.exports=bodyParser;