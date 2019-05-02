import random
import csv


# 提取随机问，同类组成正例，异类组成负例，正:负=1:3
with open('final_regroup.csv', 'w', newline='') as train:
    writer = csv.writer(train)
    with open('final_syn_train.csv', 'r') as zhidao:
        reader = csv.reader(zhidao)
        cluster = []
        cur = []
        stand = ''
        # 将同一标准问的随机问组成一个数组
        for line in reader:
            if line[1] == stand:
                cur.append(line[0])
            else:
                if cur:
                    cluster.append(cur)
                stand = line[1]
                cur = [line[0]]
        cluster.append(cur)

        # 遍历每个分类中的每个句子，在同类数组中取一条数据组成正例，在异类数组中取3条数据组成反例
        for i in range(len(cluster)):
            for j in range(len(cluster[i])):
                k = random.randint(0, len(cluster[i])-1)
                writer.writerow([cluster[i][j], cluster[i][k], 1])
                m = n = 0
                for _ in range(3):
                    while m == i:
                        m = random.randint(0, len(cluster)-1)
                    n = random.randint(0, len(cluster[m])-1)
                    writer.writerow([cluster[i][j], cluster[m][n], 0])


# 提取随机问，与正确标准问组成正例，与非正确标准问组成负例，正:负=1:3 （此方法效果更好）
with open('final_regroup.csv', 'w', newline='') as train:
    writer = csv.writer(train)
    with open('standard.csv', 'r') as standard:
        reader = csv.reader(standard)
        stand = []
        for line in reader:
            stand.append(line[0])
    with open('final_syn_train.csv', 'r', encoding='gbk') as zhidao:
        reader = csv.reader(zhidao)
        for line in reader:
            writer.writerow([line[0], line[1], 1])
            for _ in range(3):
                k = random.randint(0, 208)
                writer.writerow([line[0], stand[k], 0])