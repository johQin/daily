#!/usr/bin/env node

/**
 * 目的：采集用户信息的命令行程序
 * 通用：每一步都要判断用户输入的合法性，不合法提示重新输入。
 * 逻辑：
 * 1. 程序开始，提示用户是否进入用户信息的录入，Y/N
 * 2. 输入用户名
 * 3. 选择性别：男（1），女（2），
 * 4. 输入生日
 * 5. 选择职业，从已给出的列表中选择，也可以直接输入职业名称
 * 6. 最后提示用户是否保存，是则保存到json文件，否则退出程序
 * 7. 使用用户名作为文件名保存到当前目录下。
*/

const readline=require('readline');
const fs=require('fs');
const rl=readline.createInterface({
    input:process.stdin,
    output:process.stdout
})
let person={
    name:null,
    gender:null,
    birthday:null,
    job:null
}
let index=0;
const list=[
    {
        question:'是否进入程序，完成信息录入工作 (y/n)?',
        handler:function(answer){
            answer=answer.trim();
            if(answer===''||answer.toLowerCase()=='y'){
                console.log('进入程序,开始录入信息');
                index++;
                console.log(list[index].question)
            }else if(answer.toLowerCase()=='n'){
                console.log('取消录入，退出程序')
                process.exit(0);
            }else{
                console.log('输入不合法，请输入(y/n)?')
            }
        }
    },
    {
        question:'name：英文字符，最长不超过20个字符',
        handler:function(answer){
            answer=answer.trim();
            const regex=/^[A-Za-z]{1,20}$/
            if(regex.test(answer)){
                person.name=answer;
                index++;
                console.log(list[index].question)
            }else{
                console.log('输入不合法')
            }
        }
    },
    {
        question:'gender：male(1) / female(2)',
        handler:function(answer){
            answer=parseInt(answer.trim());
            if(answer==1||answer==2){
                person.gender=answer;
                index++;
                console.log(list[index].question)
            }else{
                console.log('输入不合法，male(1) / female(2)')
            }
        }
    },
    {
        question:'birthday，eg：2020-09-03',
        handler:function(answer){
            answer=answer.trim();
            const regex=/^[0-9]{4}-[0-9]{2}-[0-9]{2}$/
            if(regex.test(answer)){
                person.birthday=answer;
                index++;
                console.log(list[index].question)
            }else{
                console.log('输入不合法，只能输入数字和-')
            }
        }
    },
    {
        question:'请选择您的职业，工程师[1]、教师[2]、作家[3]、其他[4]',
        handler:function(answer){
            answer=parseInt(answer.trim());
            if(0<answer&& answer<5){
                person.job=answer;
                index++;
                console.log(list[index].question)
            }else{
                console.log('输入不合法，只能输入1-4')
            }
        }
    },
    {
        question:'请确认是否保存个人信息，(y/n)?',
        handler:function(answer){
            answer=answer.trim();
            if(answer===''||answer.toLowerCase()=='y'){
                fs.writeFile(`./${person.name}.json`,JSON.stringify(person),(err)=>{
                    if(err) return console.log(err.message);
                    console.log('个人信息录入成功');
                    process.exit(0);
                })
            }else if(answer.toLowerCase()=='n'){
                console.log('不保存个人信息，退出程序')
                process.exit(0);
            }else{
                console.log('输入不合法，请输入(y/n)?')
            }
        }
    },

]
rl.on('line',(input)=>{
    list[index].handler(input)
})
console.log('欢迎使用用户个人信息采集程序!');
console.log(list[index].question)
rl.resume();//如果 input 流已暂停，则 rl.resume() 方法将恢复它。


