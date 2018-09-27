import os

home = os.getenv("HOME")
root = home + "/datasets/Rumor/rumdect/Weibo/"
# root = "simple_dataset/"
tmp_path = 'tmp/'
label = home + '/datasets/Rumor/rumdect/Weibo.txt'
vocab_path = './vocab.pkl'
batch_size = 1
vec_path = home + "/datasets/WordVec/sgns.weibo.word"
dict_path = './dict.pkl'

max_document_len = 60
max_reply_len = 20
max_user_description_len = 20
max_verified_reason_len = 8

epoch = 1

# model
embed_dim = 300
hidden_dim = 100

# vocabulary
vocab_size = 50000

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
pad_idx = 0
unk_idx = 1

if __name__ == '__main__':
    print(root)
