#!/usr/bin/bash
back_dir=./mysql_back
if test ! -d $back_dir;then
    mkdir -p $back_dir
fi

echo "${back_dir}创建成功"

# 第一次执行
# $ bash -vx 判断文件是否存在.sh
# #!/usr/bin/bash
# back_dir=./mysql_back
# + back_dir=./mysql_back
# if test ! -d $back_dir;then
#     mkdir -p $back_dir
# fi
# + test '!' -d ./mysql_back
# + mkdir -p ./mysql_back

# echo "${back_dir}创建成功"
# + echo ./mysql_back创建成功
# ./mysql_back创建成功

# 第二次执行
# $ bash -vx 判断文件是否存在.sh
# #!/usr/bin/bash
# back_dir=./mysql_back
# + back_dir=./mysql_back
# if test ! -d $back_dir;then
#     mkdir -p $back_dir
# fi
# + test '!' -d ./mysql_back

# echo "${back_dir}创建成功"
# + echo ./mysql_back创建成功
# ./mysql_back创建成功
