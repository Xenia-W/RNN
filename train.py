
from utils.dataset import Dataset
from utils.word_embedding import build_embedding_matrix
import config
from model import Model
from torch import optim
import time


class Train():
    def __init__(self):
        self.dataset = Dataset()
        self.dataset_test = Dataset(mode="test")
        self.embedding_matrix = build_embedding_matrix(self.dataset.vocab.word2id,config.embed_dim)
        self.model = Model()
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self):
        start_time = time.time()
        event_generator = self.dataset.events()
        while True:
            try:
                event = next(event_generator)
            except:
                print('This epoch is done.')
                break
            self.optimizer.zero_grad()
            output, loss, result = self.model(event)
            print(loss, 'time:', time.time() - start_time, "line length:", event.len_reply)
            start_time = time.time()
            loss.backward()
            self.optimizer.step()

        # test
        event_num = 0
        true_num = 0
        test_event_generator = self.dataset_test.events()
        while True:
            try:
                event = next(test_event_generator)
                event_num += 1
            except:
                print('This epoch is done.')
                break
            output, loss, result = self.model(event)
            print(loss)
            if result >= 0.5:
                result = 1
            else:
                result = 0
            if result == event.label:
                true_num += 1
        print(true_num/event_num)




if __name__ == '__main__':
    train = Train()
    for i in range(0,config.epoch):
        train.train()
    # print(train.dataset.vocab.vocab_size)
    # print(len(train.embedding_matrix))


