let a={name:'qin'};
let b={name:'kang',age:12}
// a=b;
// b=a;
type container<t>={value:t};
let c:container<string>={value:'nihao'}

type test={name:string}|undefined;

let row:test;
console.log(row?.name)

interface Person {
    name: string;
    age: number;
  }
  
  const person = {} as Person;
  person.name = 'John';
  console.log(person.age)

type na=string|number;
type c=keyof na
let g:c=false;
  