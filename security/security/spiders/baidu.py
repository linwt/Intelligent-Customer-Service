# -*- coding: utf-8 -*-

import scrapy
from security.items import SecurityItem
import csv
import time

class BaiduSpider(scrapy.Spider):
    name = 'baidu'
    allowed_domains = ['www.baidu.com']
    start_urls = []

    # 读取标准问，拼接链接
    with open('standard.csv') as sd:
        rows = csv.reader(sd)
        for row in rows:
            start_urls += ['http://www.baidu.com/s?wd=' + row[0]]

    def parse(self, response):
        # 定位所有标题
        contents = response.xpath('//h3')
        for content in contents:
            item = SecurityItem()
            # 获取标题内容
            question = content.xpath('./a//text()').extract()
            question = ''.join(question)
            if '_' in question:
                item['question'] = question[:question.index('_')].strip()
            elif '-' in question:
                item['question'] = question[:question.index('-')].strip()
            elif '—' in question:
                item['question'] = question[:question.index('—')].strip()
            else:
                item['question'] = question
            # 获取标准问内容
            standard = response.xpath('//title/text()').extract_first()
            item['standard'] = standard[:standard.index('_')].strip()
            yield item

        # 上一页和下一页标签属性相同，需判断并获取下一页链接
        links = response.xpath("//div[@id='page']/a[@class='n']")
        for link in links:
            if link.xpath('./text()').extract_first() == '下一页>':
                next = link.xpath("./@href").extract_first()
                url = response.urljoin(next)
                # 防止爬取速度太快被禁止
                time.sleep(3)
                yield scrapy.Request(url=url, callback=self.parse)
