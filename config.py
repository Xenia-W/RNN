import torch
import argparse
import os
home = os.getenv("HOME")

class Config(object):
    ## Misc
    # opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)  # 这种适合 argparse 中传入
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tmp_path = "./tmp/"  # 所有中间处理的结果都放在这里


    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    pad_idx = 0
    unk_idx = 1



    label = home + '/datasets/Rumor/rumdect/Weibo.txt'
    vocab_path = './tmp/vocab.pkl'
    vec_path = home + "/datasets/WordVec/sgns.weibo.word"
    dict_path = './dict.pkl'


    max_document_len = 60
    max_reply_len = 20
    max_user_description_len = 20
    max_verified_reason_len = 8


opt = Config()  # 只有主文件调用, 其他文件通过主文件传递过去; 因为 opt 内容有时会改

parser = argparse.ArgumentParser(description='JARVIS')
## learning
parser.add_argument('-epoch', type=int, default=8, help='number of epochs for train [default: 300]')
parser.add_argument('-batch-size', type=int, default=1, help='batch size for training [default: 64]')

parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension, will auto load Glove vector of that dim [default: 100], alternative: [25, 50, 100, 200]')
parser.add_argument('-hidden-dim', type=int, default=200, help='number of hidden dimension [default: 256]')
## vocabulary
parser.add_argument('-vocab-size', type=int, default=50000, help='vocabulary size of this model [default: 50000]')  # 非 None, 且小于语料原本的 vocab_size 时, 会减小语料词典大小
## device
parser.add_argument('-device-id', type=int, default=1, help='device to use for iterate data, -1 mean cpu [default: 1]')
## view
parser.add_argument('-reply', action='store_true', default=False, help='device to use for iterate data, -1 mean cpu [default: 1]')
parser.add_argument('-profile', action='store_true', default=False, help='device to use for iterate data, -1 mean cpu [default: 1]')

## small
parser.add_argument('-small', action='store_true', default=False, help='device to use for iterate data, -1 mean cpu [default: 1]')
## save path
parser.add_argument('-save-dir', type=str, default="./multiview/", help='device to use for iterate data, -1 mean cpu [default: 1]')

args = parser.parse_args()

if args.small:
    args.root = "simple_dataset/"
else:
    args.root = home + "/datasets/Rumor/rumdect/Weibo/"

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


if opt.device == torch.device('cuda'):
    torch.cuda.set_device(args.device_id)

