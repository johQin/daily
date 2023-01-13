#!/usr/bin/bash
ip=www.baidu.com
ping -c1 $ip &> /dev/null && echo "${ip} is up" || echo "${ip} is down"