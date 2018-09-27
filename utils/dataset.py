import glob
import random

import config
from utils.vocabulary import Vocabulary
# from utils.batch import Batch
from utils.event import Event
random.seed(100)

class Dataset:
    def __init__(self, mode="train"):
        self.vocab = Vocabulary()
        self.mode = mode

    def events(self):
        file_list = glob.glob(config.root + "*.json")
        random.shuffle(file_list)

        with open(config.label) as f:
            d = {}
            for line in f:
                s = line.split("	")
                idx = s[0].split(":")[1]
                label = s[1].split(':')[1]
                d[idx] = label
        if self.mode == "train":
            data_set = file_list[:int(0.675*len(file_list))]
        elif self.mode == "test":
            data_set = file_list[int(0.225 * len(file_list)):]
        # val_set = file_list[int(0.7*len(file_list)):int(0.8*len(file_list))]


        for i in range(len(data_set))[::config.batch_size]:
            file = data_set[i]
            label = d[file.split('.')[0].split('/')[-1]]
            yield Event(file, int(label), self.vocab)


if __name__ == '__main__':
    dataset = Dataset()
    dataset.batches()





