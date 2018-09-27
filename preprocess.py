import config
import os

from utils.dataset import Dataset
from utils.word_embedding import build_embedding_matrix

if not os.path.exists(config.tmp_path):
    os.makedirs(config.tmp_path)
if not os.path.exists(config.dict_path):
    os.makedirs(config.dict_path)
dataset = Dataset()
build_embedding_matrix(dataset.vocab.word2id, config.embed_dim)

if __name__ == '__main__':  # train.py 中也可以调用 prepocess.py
    print("词典大小:", dataset.vocab.vocab_size)


