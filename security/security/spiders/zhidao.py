# -*- coding: utf-8 -*-

import scrapy
from security.items import SecurityItem
import csv
import time

# 执行爬虫并保存文件： scrapy crawl zhidao -o zhidao.csv

class ZhidaoSpider(scrapy.Spider):
    name = 'zhidao'
    allowed_domains = ['zhidao.baidu.com']
    start_urls = []

    # 读取标准问，拼接链接
    with open('standard.csv') as sd:
        rows = csv.reader(sd)
        for row in rows:
            start_urls += ['http://zhidao.baidu.com/search?word=' + row[0]]
    # print(start_urls)

    def parse(self, response):
        # 定位所有标题
        contents = response.xpath('//dl')
        for content in contents:
            item = SecurityItem()
            # 获取标题内容
            question = content.xpath('./dt//a//text()').extract()
            item['question'] = ''.join(question)
            # 获取标准问内容
            standard = response.xpath('//title/text()').extract_first()
            item['standard'] = standard[standard.index('_')+1:].strip()
            yield item

        # 获取下一页链接
        next = response.xpath("//div[@class='pager']/a[@class='pager-next']/@href").extract_first()
        url = response.urljoin(next)
        time.sleep(3)
        yield scrapy.Request(url=url, callback=self.parse)