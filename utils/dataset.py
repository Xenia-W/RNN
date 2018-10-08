import glob
import random

from config import opt
from utils.vocabulary import Vocabulary
# from utils.batch import Batch
from utils.event import Event
random.seed(100)

class Dataset:
    def __init__(self, args, mode="train"):
        self.args = args
        self.vocab = Vocabulary(self.args)
        self.mode = mode

    def events(self):
        file_list = glob.glob(self.args.root + "*.json")
        random.shuffle(file_list)

        with open(opt.label) as f:
            d = {}
            for line in f:
                s = line.split("	")
                idx = s[0].split(":")[1]
                label = s[1].split(':')[1]
                d[idx] = label
        if self.mode == "train":
            data_set = file_list[:int(0.675*len(file_list))]
        elif self.mode == "test":
            data_set = file_list[int(0.675 * len(file_list)):]
        # val_set = file_list[int(0.7*len(file_list)):int(0.8*len(file_list))]


        for i in range(len(data_set))[::self.args.batch_size]:
            file = data_set[i]
            label = d[file.split('.')[0].split('/')[-1]]
            yield Event(self.args, file, int(label), self.vocab)


if __name__ == '__main__':
    dataset = Dataset()
    dataset.batches()





