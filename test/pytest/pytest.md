# Pytest

# 0 初识

## 0.1 命名规范

- 模块名（py文件）必须是以`test_`开头或者`_test`结尾
- 测试类（class）必须以Test开头（大驼峰），并且不能带init方法，类里的方法必须以`test_`开头
- 测试用例（函数）必须以`test_`开头



# 1 Fixture

Fixture（夹具），在用例之前，执行之后，自动运行代码

```python
import pytest
from selenium import webdriver

@pytest.fixture
def browser():
    # 用例之前
    driver = webdriver.Chrome()
    driver.get("https://www.baidu.com")
    
    yield dirver
    
    # 用例之后
    driver.quit()
```

fixture中使用yield还是return	

- 只需要 “前置准备”，不需要 “后置清理” → 用 `return`
  - 只提供**数据、对象、配置**

- **既需要 “前置准备”，又需要 “后置清理” → 必须用 `yield`**



## 1.1 请求fixture

两种方式：

- 直接作为测试函数的参数

  ```python
  import pytest
  
  @pytest.fixture
  def db_connection():
      print("\n[fixture] 连接数据库")
      conn = "模拟数据库连接对象"
      yield conn  # yield 之前是前置，之后是后置
      print("\n[fixture] 关闭数据库连接")
  
  def test_db_query(db_connection):
      print(f"[test] 使用连接：{db_connection}")
      assert db_connection is not Non
  ```

- 使用`@pytest.mark.usefixture`装饰器（不关心返回值时使用）

  ```python
  import pytest
  
  @pytest.fixture
  def setup_environment():
      print("\n[fixture] 初始化测试环境")
      yield
      print("\n[fixture] 清理测试环境")
  
  # 不接收返回值，只需要执行fixture的前后置
  @pytest.mark.usefixtures("setup_environment")
  def test_something():
      print("[test] 执行测试逻辑")
      assert 1 + 1 == 2
  ```

## 1.2 作用域

控制 fixture 的创建和销毁时机，避免重复创建资源。

可选值：`function`（默认）<  `class`  <  `module`  <  `session`

session:

- 整个 pytest 命令启动到结束，只执行 1 次前置
- 所有测试全部跑完后，只执行 1 次后置
- 有测试用例共享同一个 fixture 对象（不重复创建）

```python
import pytest

# scope="function"：每个测试函数执行前后各一次（默认）
@pytest.fixture(scope="function")
def func_scope():
    print("\n[func_scope] 初始化")
    yield
    print("\n[func_scope] 销毁")

# scope="module"：整个模块（文件）只初始化/销毁一次
@pytest.fixture(scope="module")
def mod_scope():
    print("\n[mod_scope] 初始化（整个文件只执行一次）")
    yield
    print("\n[mod_scope] 销毁（整个文件只执行一次）")

def test_func1(func_scope, mod_scope):
    print("[test_func1] 执行")

def test_func2(func_scope, mod_scope):
    print("[test_func2] 执行")
```



## 1.3 fixture 参数化

用 `params` 参数给 fixture 传入多组数据，让依赖它的测试自动执行多轮。

```python
import pytest

# fixture 参数化：提供多组用户数据
@pytest.fixture(params=[
    {"name": "windy", "age": 25},
    {"name": "张三", "age": 30},
    {"name": "李四", "age": 18}
])
def user_info(request):
    # request.param 拿到当前这组参数
    return request.param

def test_user_age(user_info):
    print(f"测试用户：{user_info['name']}，年龄：{user_info['age']}")
    # 年龄必须是正整数
    assert user_info["age"] > 0
```

运行：`pytest -v test_demo.py`

- 这个测试会自动执行 **3 次**，分别用 fixture 里的 3 组参数。

## 1.4 共享fixture

把公共的 fixture 放在 `conftest.py` 里，**同目录下**的所有测试文件都能直接用，不需要 import。

```python
# conftest.py
import pytest

@pytest.fixture(scope="session")
def global_config():
    print("\n[conftest] 加载全局配置（整个会话只一次）")
    return {"env": "test", "timeout": 10}

@pytest.fixture
def clean_temp_files():
    print("\n[conftest] 清理临时文件")
    yield
    print("\n[conftest] 测试后删除临时文件")


    
    
# 同目录下，不同py文件
# test_case1.py
def test_case1(global_config, clean_temp_files):
    print(f"用全局配置：{global_config}")
    assert global_config["env"] == "test"

# test_case2.py
def test_case2(global_config):
    assert global_config["timeout"] > 0
```

## 1.5 其他参数

- autouse
  - 每个测试函数都会自动调用该fixture,无需传入fixture函数名。
  - 默认False，autouse=True自动调用，无需传仍何参数，作用范围跟着scope走（谨慎使用）
- name
  - fixture的重命名，如果使用了name,那只能将name传如，函数名不再生效

## 1.6 fixture 常见错误

- 小 scope 可以依赖大 scope，但不能相反。
-  fixture 名和测试函数参数名冲突。参数名和 fixture 同名，但又被其他 fixture 覆盖 / 重名。参数名和 fixture 名都是 xxx，没问题，但如果同名变量多了容易混乱。
- 在 fixture 里做复杂业务逻辑，而不是只做资源准备





