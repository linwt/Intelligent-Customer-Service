import csv
import random
import jieba

# 随机打乱词序
with open('final_shuffle.csv', 'w', newline='') as shuffle:
    writer = csv.writer(shuffle)
    with open('compete_v1.csv', 'r') as v1:
        reader = csv.reader(v1)
        part_rand = []
        stand = ''
        for line in reader:
            # 如果是最后一个标准问会跳过同义词扩充部分，所以需要在文件最后一行随便加一行不同标准问的数据
            if line[1] == stand:
                part_rand.append(line[0])
            else:
                for rand in part_rand:
                    cut_word = [word for word in jieba.cut(rand)]
                    len_cut_word = len(cut_word)
                    writer.writerow([rand, stand])
                    for time in range(120 // len(part_rand)):
                        random.shuffle(cut_word)
                        writer.writerow([''.join(cut_word), stand])
                part_rand = [line[0]]
                stand = line[1]