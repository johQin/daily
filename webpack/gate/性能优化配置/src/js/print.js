console.log('打印文件被加载了');
function print(content) {
  console.log(content);
}
export function mul(x, y) {
  console.log(`乘法的结果${x * y}`);
  return x * y;
}
export function sub(x, y) {
  console.log(`减法的结果${x - y}`);
  return x - y;
}
export default print;
