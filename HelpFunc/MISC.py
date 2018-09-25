# last time modified: 2018/6/18
# 一些混杂但是好用的函数

import time


# 1) 报时函数
def get_now_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


# 2) 像平常用的rep一样
def rep(data, each=3):
    result = []
    for d in data:
        for i in range(each):
            result.append(d)
    return result
