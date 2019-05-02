# coding=gbk

from sklearn.model_selection import train_test_split
import csv
import random
import synonyms
import pandas as pd


# 同义词随机替换
with open('final_syn.csv', 'w', newline='') as expand:
    writer = csv.writer(expand)
    with open('final_shuffle.csv', 'r', encoding='gbk') as all:
        reader = csv.reader(all)
        part_rand = []
        stand = ''
        for line in reader:
            # 将同个标准问的随机问添加到一个数组
            # 如果是最后一个标准问会跳过同义词扩充部分，所以需要在文件最后一行随便加一行不同标准问的数据
            if line[1] == stand:
                part_rand.append(line[0])
            else:
                for rand in part_rand:
                    # 将原句写入
                    writer.writerow([rand, stand])
                    # 句子分词
                    cut_word = synonyms.seg(rand)
                    syns = []
                    # 获取每个词的十个同义词，添加到数组
                    for word in cut_word[0]:
                        syn = synonyms.nearby(word)
                        syns.append(syn[0])
                    # 由已有数据量确定每个句子扩充几遍
                    # for i in range(250//len(part_rand)):
                    for i in range(1):
                        new = ''
                        # 遍历每个词，增加替换概率，0.6不变、0.2替换、0.2unk
                        for index, word in enumerate(cut_word[0]):
                            syn = syns[index]
                            k = random.randint(0, 9)
                            if k in range(6):
                                new += word
                            elif k in range(6, 8):
                                if not syn:
                                    new += word
                                    continue
                                new += syn[random.randint(1, 9)]
                            else:
                                new += ''
                        # 写入文件
                        writer.writerow([new, stand])
                # 准备添加下一组随机问
                part_rand = [line[0]]
                stand = line[1]
			
			
# 读取全部数据集，随机划分成训练集和测试集
data = pd.read_csv('final_syn.csv', encoding='gbk')
train, test = train_test_split(data, test_size=0.05)
train.to_csv('final_syn_train.csv', header=False, index=False)
test.to_csv('final_test.csv', header=False, index=False)