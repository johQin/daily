function classDecorator(params:any){
    return function(target:any){
        console.log('类装饰器'+params)
    }
}
function propertyDecorator(params:any){
    return function(target:any,attr:any){
        console.log('属性装饰器'+params)
    }
}
function methodDecorator(params:any){
    return function(target:any,methodName:any,desc:any){
        console.log('方法装饰器'+params)
    }
}
function paramDecorator(params:any){
    return function(target:any,methodName:any,paramsIndex:any){
        console.log('方法参数装饰器',paramsIndex)
    }
}
@classDecorator(1)
@classDecorator(2)
class HttpClient{
    @propertyDecorator(1)
    url:string|undefined;
    @methodDecorator(2)
    setData(val:string):void{
        console.log(val)
    }
    @methodDecorator(1)
    @methodDecorator(11)
    getData(@paramDecorator(1) p1:any,@paramDecorator(2) p2:any):void{
        console.log(p1,p2)
    }
    @propertyDecorator(2)
    uuid:string|undefined;

}
let h =new HttpClient();
