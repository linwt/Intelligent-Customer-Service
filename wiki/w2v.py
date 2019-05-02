import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# 模型训练和保存：分词
model = Word2Vec(LineSentence('wiki_cut_word.txt'), size=300, window=5, min_count=5, workers=multiprocessing.cpu_count(), iter=100)
model.save('wiki_word_model')
model.wv.save_word2vec_format('wiki_word_vector', binary=False)

# 模型加载和测试
model = Word2Vec.load('wiki_word_model')
print(model.most_similar('股票'))
print(model['三板'])
print(model.doesnt_match("共同 文化 丰富 数学".split()))
print(model.wv.syn0.shape)
print(model.wv.syn0)
print(model.wv.index2word)
print(model.wv.index2word[25])
word2index = {v:k for k,v in enumerate(model.wv.index2word)}
print(word2index['数学'])


# 模型训练和保存：分字
model = Word2Vec(LineSentence('wiki_cut_char.txt'), size=300, window=5, min_count=5, workers=multiprocessing.cpu_count(), iter=100)
model.save('wiki_char_model')
model.wv.save_word2vec_format('wiki_char_vector', binary=False)

# 模型加载和测试
model = Word2Vec.load('wiki_char_model')
print(model.most_similar('学'))
print(model['学'])
print(model.doesnt_match("学 理 经 好".split()))
print(model.wv.syn0.shape)
print(model.wv.syn0)
print(model.wv.vectors)
print(model.wv.index2word)
print(model.wv.index2word[25])
word2index = {v:k for k,v in enumerate(model.wv.index2word)}
print(word2index['学'])
