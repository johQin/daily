package interpreter;

import java.util.HashMap;


/**
 * 变量的解释器
 *
 */
public class VarExpression extends Expression {

    private String key; // key=a,key=b,key=c

    public VarExpression(String key) {
        this.key = key;
    }

    // 这里的函数参数var ：{a=10, b=20}
    // interpreter 根据 变量名称，返回对应值
    @Override
    public int interpreter(HashMap<String, Integer> var) {
        return var.get(this.key);
    }
}
