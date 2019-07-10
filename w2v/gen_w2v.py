import logging
import gensim
from gensim.models import word2vec

# wordVec = gensim.models.KeyedVectors.load_word2vec_format("word2Vec.bin", binary=True)

sentences = word2vec.LineSentence("F:\git_pro/data/preProcess/wordEmbdiing.txt")

# word2vec模型
model = gensim.models.Word2Vec(sentences, size=200, sg=1, iter=8)

#保存模型
model.wv.save_word2vec_format("word2Vec.bin", binary=True)