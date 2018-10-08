import numpy as np
import glob

from config import opt, args
from utils.dataset import Dataset
from utils.vocabulary import Vocabulary

def preview_args(args):
    print('========== Begin something about args')

    print('Max Vocab Size:', args.vocab_size)
    print('Hidden Size:', args.hidden_dim)
    print('Embedding Dim:', args.embed_dim)
    print()
    print('Epoch:', args.epoch)
    print('Device:', opt.device)
    print('Device ID:', args.device_id)
    print()
    print('Batch Size', args.batch_size)
    print("Reply", args.reply)
    print("Profile", args.profile)
    print("Small", args.small)
    print('========== End something about args')

def model_parameters():
    from train import Train
    model = Train().model

    print('========== Begin someting about model parameters:')
    print("Encoder Model Parameters:", sum(p.nelement() for p in model.encoder.parameters()))
    print("Decoder Model Parameters:", sum(p.nelement() for p in model.decoder.parameters()))
    print("Reduce State Model Parameters:", sum(p.nelement() for p in model.reduce_state.parameters()))
    print("Total Model Parameters:", sum(p.nelement() for p in model.parameters()))
    print()

def dataset():
    vocab = Vocabulary(args)
    dataset = Dataset(args, vocab)
    source_files = sorted(glob.glob(args.dataset_file_path + 'train_source*.dat'))
    target_files = sorted(glob.glob(args.dataset_file_path + 'train_target*.dat'))

    print('========== Begin someting about vocabulary:')
    print('Vocab Size:', dataset.vocab.vocab_size)
    print('First 10 Word2cnt:', list(dataset.vocab._word2cnt.items())[:10])
    print()

    print('========== Begin someting about dataset:')
    X_lens = [len(sen.split()) for source_file in source_files for sen in open(source_file)]
    y_lens = [len(sen.split()) for target_file in target_files for sen in open(target_file)]
    print('Number of Source Sentences:', len(X_lens))
    print('Number of Sarget Sentences:', len(y_lens))
    print()
    print('Mean Length of Source Sentences:', np.mean(X_lens))
    print('Max Length of Source Sentences:', np.max(X_lens))
    print('Min Length of Source Sentences:', np.min(X_lens))
    print()
    print('Mean Length of Target Sentences:', np.mean(y_lens))
    print('Max Length of Target Sentences:', np.max(y_lens))
    print('Min Length of Target Sentences:', np.min(y_lens))
    print()

if __name__ == '__main__':
    preview_args(args)
    model_parameters()
    dataset()

## Gigaword
# Number of source sentences: 3803957
# Number of target sentences: 3803957

# Mean length of source sentences: 31.5213665664
# Max length of source sentences: 101
# Min length of source sentences: 11

# Mean length of target sentences: 8.2792174044
# Max length of target sentences: 45
# Min length of target sentences: 2

## Byte Cup 2018
# Number of Source Sentences: 1183301
# Number of Sarget Sentences: 1183301

# Mean Length of Source Sentences: 633.256423345
# Max Length of Source Sentences: 36712
# Min Length of Source Sentences: 0

# Mean Length of Target Sentences: 11.8644385494
# Max Length of Target Sentences: 57
# Min Length of Target Sentences: 0

## CNN/DM Processed
# Number of Source Sentences: 287227
# Number of Sarget Sentences: 287227

# Mean Length of Source Sentences: 791.384664394
# Max Length of Source Sentences: 2882
# Min Length of Source Sentences: 0

# Mean Length of Target Sentences: 62.7471198738
# Max Length of Target Sentences: 2352
# Min Length of Target Sentences: 6
