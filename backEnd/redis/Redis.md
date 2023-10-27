# Redis

# log

1. Redis的URL写法格式如下：

   - **redis://[:password@]host[:port][/db_number][/db_number][?option=value]**

   - ```bash
     # 可以只有密码没有用户
     redis://:123456@192.168.100.138:6379/10
     # db在后面指定
     redis://:123456@192.168.100.138:6379?timeout=3000&db=10&max_connections=50
     ```

   - 