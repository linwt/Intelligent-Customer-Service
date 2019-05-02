# -*- coding: utf-8 -*-

import scrapy
from security.items import SecurityItem
import csv
import time

class SogouSpider(scrapy.Spider):
    name = 'sogou'
    allowed_domains = ['www.sogou.com']
    start_urls = []

    # 读取标准问，拼接链接
    with open('standard.csv') as sd:
        rows = csv.reader(sd)
        for row in rows:
            start_urls += ['http://www.sogou.com/sogou?query=' + row[0]]

    def parse(self, response):
        # 定位所有标题
        contents = response.xpath('//h3')
        for content in contents:
            item = SecurityItem()
            # 获取标题内容
            question = content.xpath('./a//text()').extract()
            question = ''.join(question)
            if '-' in question:
                item['question'] = question[:question.index('-')].strip()
            elif '_' in question:
                item['question'] = question[:question.index('_')].strip()
            else:
                item['question'] = question
            # 获取标准问内容
            standard = response.xpath('//title/text()').extract_first()
            item['standard'] = standard[:standard.index('-')].strip()
            yield item

        # 获取下一页链接
        next = response.xpath("//div[@class='p']/a[@class='np']/@href").extract_first()
        url = response.urljoin(next)
        time.sleep(5)
        yield scrapy.Request(url=url, callback=self.parse)

