import os
import numpy as np
import pickle

from config import opt


def load_word_vec(path=opt.vec_path, word2idx=None):  # 只加载语料 vocab 中出现过的词
    assert os.path.exists(path)
    with open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        word2vec = {}
        for line in list(fin)[1:]:
            tokens = line.rstrip().split()  # rstrip 防止最左侧奇怪字符
            if word2idx is None or tokens[0] in word2idx.keys():
                word2vec[tokens[0]] = np.asarray(tokens[1:], dtype='float64')
    return word2vec


def build_embedding_matrix(word2idx,embed_dim=300, d_type='dataset'):  # 按 vocab 顺序保存 embedding_matrix
    embedding_matrix_file_name = './tmp/{0}_{1}_embedding_matrix.dat'.format(d_type, str(embed_dim))
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0-3 are all-zeros
        fname = opt.vec_path
        word2vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word2vec.get(word)
            if vec is not None:  # words not found in embedding index will be all-zeros.
                if (len(vec)) == 300:
                    embedding_matrix[i] = vec

        print('saving embedding_matrix')
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix

if __name__ == '__main__':
    # 测试: load_all_word_vec 所有的 word vec
    all_word2vec = load_word_vec()
    print(len(all_word2vec.keys()))  # wiki_6b: 400001; twitter_27b: 1193515
    print(all_word2vec['，'])  # (100,)

    # 测试: load_word_vec 语料中出现的 word vec
    # from utils.dataset import Dataset
    # dataset = Dataset()
    # dataset.load_dataset()
    # print(len(dataset.vocab._word2idx))  # 58
    # word2vec = load_word_vec(word2idx=dataset.vocab._word2idx)
    # print(word2vec.keys())  # dict_keys(['the', ',', '.', 'and', 'in', 'is', 'it', 'his', 'but', 'new', 'united', 'during', 'states', 'our', 'my', 'march', 'never', 'least', 'june', 'july', 'your', 'april', 'september', 'january', 'december', 'november', 'california', 'paris', 'sometimes', 'usually', 'spring', 'hot', 'jersey', 'cold', 'favorite', 'orange', 'apple', 'warm', 'quiet', 'fruit', 'busy', 'liked', 'autumn', 'mild', 'freezing', 'lemon', 'grape', 'chilly', 'relaxing', 'snowy'])
    #
    # # 测试: build_embedding_matrix
    # embedding_matrix = build_embedding_matrix(dataset.vocab._word2idx, opt.embed_dim)
    # print(embedding_matrix.shape)  # (60, 100)
    # print(embedding_matrix[-6])
    # print(dataset.vocab._word2idx.keys())
