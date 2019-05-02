# 演示视频
链接：https://pan.baidu.com/s/1SfywE5AoKXF3e9IyjeECvg   
提取码：jkor 
# 文件说明
* data：包括爬虫数据、扩充数据、官方数据
* security：爬取百度、百度知道、搜狗数据
* wiki：获取维基百科数据进行分词和分字处理，并训练词向量和字向量模型
* process：对爬虫数据和官方数据进行处理
* model：单个强模型，微调得到多个弱模型，投票方式融合
# 获取维基百科数据
一、维基百科数据下载地址  
&emsp; https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2   
二、开源解压项目  
&emsp; https://github.com/attardi/wikiextractor   
&emsp; 1、直接复制WikiExtractor.py文件即可  
&emsp; 2、解压文件E:\wikiextractor>python WikiExtractor.py -cb 1500M -o extracted E:\zhwiki-latest-pages-articles.xml.bz2  
&emsp; 3、得到E:\wikiextractor\extracted\AA\wiki_00.bz2，解压wiki_00.bz2得到wiki_00，重命名为wiki.txt  
三、下载opencc  
&emsp; 1、下载opencc windows版  
&emsp; 2、将bin目录路径添加到环境变量  
四、简繁体转换  
&emsp; E:\wiki\extracted\AA> opencc -i wiki.txt -o wiki_jian.txt -c E:\wiki\opencc-1.0.4-win32\opencc-1.0.4\share\opencc\t2s.json  
五、分词、分字  
&emsp; 将wiki_jian.txt按照分词和分字两种方法进行切分，并保存到txt文件中
# 模型指标
单模型|正确率|召回率|F1值
--|--|--|--|
模型1|0.862|0.767|0.812
模型2|0.859|0.758|0.805
模型3|0.964|0.370|0.535
模型4|0.931|0.570|0.707
模型5|0.924|0.611|0.735

融合效果|正确率|召回率|F1值
--|--|--|--|
top1(sim>0.8)|0.895|0.812|0.851
top5(sim>0.6)|0.984|0.962|0.973
