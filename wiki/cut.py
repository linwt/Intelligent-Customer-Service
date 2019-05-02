import jieba
import csv

# security分词：仅保留汉字、数字、字母
new_words = '新三板 逆回购 三方存管 中签 打新 港股通 深港通 沪港通 上证基金通 融资股'.split()
for word in new_words:
    jieba.add_word(word)
with open('security_cut_word.txt', 'w', encoding="utf8") as security_word:
    with open('dataset_all.csv', 'r') as dataset:
        for line in dataset:
            security_word.write(' '.join([word for word in jieba.cut(line)
                                          if word and (0x4E00 <= ord(word[0]) <= 0x9FA5 or word[0].isalpha() or word[0].isdigit())]) + '\n')
            print(' '.join([word for word in jieba.cut(line)
                            if word and (0x4E00 <= ord(word[0]) <= 0x9FA5 or word[0].isalpha() or word[0].isdigit())]))

# security分字
with open('security_cut_char.txt', 'w', encoding="utf8") as security_char:
    with open('dataset_all.csv', 'r') as dataset:
        for line in dataset:
            security_char.write(' '.join([char for char in line
                                       if char and (0x4E00 <= ord(char) <= 0x9FA5 or char.isalpha() or char.isdigit())]) + '\n')
            print(' '.join([char for char in line
                            if char and (0x4E00 <= ord(char) <= 0x9FA5 or char.isalpha() or char.isdigit())]))



# wiki分词：仅保留汉字、字母、数字
with open('wiki_cut_word.txt', 'w', encoding='utf8') as wiki_word:
    with open('./extracted/AA/wiki_jian.txt', 'r', encoding='utf8') as wiki_jian:
        for line in wiki_jian:
            wiki_word.write(' '.join([word for word in jieba.cut(line.strip()) if word and (0x4E00 <= ord(word[0]) <= 0x9FA5 or word[0].isalpha() or word[0].isdigit())]) + '\n')
            print(' '.join([word for word in jieba.cut(line.strip()) if word and (0x4E00 <= ord(word[0]) <= 0x9FA5 or word[0].isalpha() or word[0].isdigit())]))

# 将security的分词追加到wiki的分词中，扩展相关数据
with open('wiki_cut_word.txt', 'a', encoding="utf8") as wiki_word:
    with open('security_cut_word.txt', 'r', encoding='utf8') as security:
        for line in security:
            wiki_word.write(line)



# wiki分字：仅保留汉字、字母、数字
with open('wiki_cut_char.txt', 'w', encoding="utf8") as wiki_char:
    with open('./extracted/AA/wiki_jian.txt', 'r', encoding="utf8") as wiki_jian:
        for line in wiki_jian:
            wiki_char.write(' '.join([char for char in line.strip() if char and (0x4E00 <= ord(char) <= 0x9FA5 or char.isalpha() or char.isdigit())]) + '\n')
            print(' '.join([char for char in line.strip() if char and (0x4E00 <= ord(char) <= 0x9FA5 or char.isalpha() or char.isdigit())]))