# -*- coding:utf8 -*-
import csv


# 将百度和搜狗搜索爬取到的结果按照包含疑问关键字进行提取
key = ['?', '什么', '怎', '吗', '如何', '哪', '呢', '是否', '是不是', '多少']

with open('baidu_extract.csv', 'w', newline='') as be:
    writer = csv.writer(be)

    with open('baidu.csv', 'r') as b:
        reader = csv.reader(b)
        for row in reader:
            for k in key:
                if k in row[0]:
                    writer.writerow(row)
                    break

with open('sogou_extract.csv', 'w', newline='') as se:
    writer = csv.writer(se)

    with open('sogou.csv', 'r') as s:
        reader = csv.reader(s)
        for row in reader:
            for k in key:
                if k in row[0]:
                    writer.writerow(row)
                    break


# 将百度搜索爬取结果和百度知道爬取结果进行汇总
with open('dataset_all.csv', 'w', newline='') as train:
    writer = csv.writer(train)

    with open('baidu_extract.csv', 'r') as baidu:
        reader = csv.reader(baidu)
        for row in reader:
            writer.writerow(row)
			
	with open('sogou_extract.csv', 'r') as baidu:
        reader = csv.reader(baidu)
        for row in reader:
            writer.writerow(row)

    with open('zhidao.csv', 'r') as zhidao:
        reader = csv.reader(zhidao)
        for row in reader:
            writer.writerow(row)