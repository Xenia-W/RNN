import config
import glob
import os
import pickle
import json
import jieba


from utils.content import Content




class Vocabulary:
    def __init__(self, build_vocab=True):
        self.word2id = {}
        self.id2word = {}
        self.word2cnt = {}
        self.vocab_size = 0
        self.add_word(config.PAD_TOKEN)
        self.add_word(config.UNK_TOKEN)

        if build_vocab:
            print("Building vocabulary from file...")
            self.build_from_file()

    def add_document(self, document):
        for sent in document:
            self.add_sentence(sent)

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2cnt:
            self.word2id[word] = self.vocab_size
            self.id2word[self.vocab_size] = word
            self.word2cnt[word] = 1
            self.vocab_size += 1
        else:
            self.word2cnt[word] += 1

    def build_from_file(self):
        if os.path.exists(config.vocab_path):
            print("Vocabulary was already built")
            self.load()
            print("Vocabulary size:", self.vocab_size)
        else:
            a = 0
            # from utils.event import Event
            file_list = glob.glob(config.root + "*.json")
            for file in file_list:
                a = a + 1
                print("Processing event:", a)
                document = [list(jieba.cut(dic['original_text'], cut_all=False)) for dic in json.loads(open(file).read())]
                self.add_document(document)
            self.reduce(config.vocab_size - 4)
            self.save()
            print("Vocabulary size:", self.vocab_size)

    def save(self):
        with open(config.vocab_path, 'wb') as f:
            pickle.dump([self.word2id, self.id2word, self.word2cnt, self.vocab_size], f)

    def load(self):
        with open(config.vocab_path, 'rb') as f:
            self.word2id, self.id2word, self.word2cnt, self.vocab_size = pickle.load(f)

    def sen2id(self, sentence):
        sen = []
        for word in sentence:
            if word not in self.word2cnt:
                sen.append(config.unk_idx)
            else:
                sen.append(self.word2id[word])
        return sen

    # def doc2id(self, document):
    #     '''
    #     :param sentence:
    #     :return:
    #     '''
    #     doc = []
    #     for doc in document:
    #         for sent in doc:
    #
    #             if sent not in self.word2cnt:
    #                 doc.append(config.unk_idx)
    #             else:
    #                 doc.append(self.word2id[sent])
    #     return doc

    def reduce(self, vocab_size):
        if vocab_size > self.vocab_size:
            print("Origin vocabulary is too small.")
        else:
            print("Original vocab size:", self.vocab_size)
            word2cnt = self.word2cnt
            self.__init__(build_vocab=False)
            words = [key for (key, value) in sorted(word2cnt.items(), key=lambda x: x[1], reverse=True)[:vocab_size]]
            for word in words:
                self.word2id[word] = self.vocab_size
                self.id2word[self.vocab_size] = word
                self.word2cnt[word] = word2cnt[word]
                self.vocab_size += 1
            print("Now original vocab size:", self.vocab_size)


if __name__ == '__main__':
    vocab = Vocabulary()

    # sentence = ['给','手机','充电','，','强大','电流','通过','手机','。','充电','给','强大','手机',',','通过']
    # vocab.add_sentence(sentence)
    # print(vocab.word2cnt)
    # vocab.reduce(5)
    # print(vocab.word2cnt)
    # print(vocab.sen2id(sentence))
    counter = 0
    vocab.build_from_file()
    vocab.save()
    # for dirpath, dirnames, filenames in os.walk(config.root):
    #     for filepath in filenames:
    #         counter = counter +1
    #         print(counter)
    #         f = open(os.path.join(dirpath, filepath), encoding="utf-8")
    #         doc = json.load(f)
    #         tweetText = ""
    #         for i in range(len(doc)):
    #               seg_list = jieba.cut(doc[i]['text'], cut_all=False)
    #               tweetText = tweetText + " ".join(seg_list)
    #         sentence = tweetText.split(" ")
    #         vocab.add_sentence(sentence)
    print(len(vocab.word2id))
    print(vocab.vocab_size)
    print(vocab.id2word[5])
    print(vocab.word2id['毛'])
