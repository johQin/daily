#!/usr/bin/bash
url=www.sina.com.cn
echo $url
# 变量的长度
echo ${#url}
# 1.删除
echo '---------------字串删除----------'
echo ${url#*.}
# 输出 sina.com.cn
echo ${url##*.}
# 输出 cn
echo ${url%.*}
# 输出 www.sina.com
echo ${url%%.*}
# 输出 www

# 2.索引和切片
echo '---------------索引和切片----------'
# 从字符串的第0个开始，切长度为5个串
echo ${url:0:5}
# 输出 www.s
# 从字符串的第5个开始，切长度为5个串
echo ${url:5:5}
# 输出 ina.c
# 从字符串的第5个开始，切到末尾
echo ${url:5}
# 输出 ina.com.cn

# 3.替换
echo '---------------替换----------'
# 替换第一个被匹配的字符
echo ${url/n/N}
#www.siNa.com.cn
# 替换所有被匹配到的字符
echo ${url//n/N}
# www.siNa.com.cN