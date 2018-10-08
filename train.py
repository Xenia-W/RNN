from torch import optim
import time
import os
import torch
import glob

from utils.dataset import Dataset
from utils.word_embedding import build_embedding_matrix
from models.model import Model
from config import opt, args
from preview import preview_args



class Train():
    def __init__(self):
        self.dataset = Dataset(args)
        self.dataset_test = Dataset(args, mode="test")
        self.embedding_matrix = build_embedding_matrix(self.dataset.vocab.word2id, args.embed_dim)
        self.model = Model(args)
        self.optimizer = optim.Adam(self.model.parameters())
        self._epoch = 0
        self._iter = 0
        self.max_acc = None
        self.load_model()

    def save_model(self):
        state = {
            'epoch': self._epoch,
            'iter': self._iter,
            'max_acc': self.max_acc,
            'state_dict': self.model.state_dict(),

        }

        name =  '-{}.pth'.format(self._epoch)
        model_save_path = os.path.join(args.save_dir, name)

        print("saving model", model_save_path)

        torch.save(state, model_save_path)

    def load_model(self):
        assert os.path.exists(args.save_dir)

        if len(glob.glob(os.path.join(args.save_dir) + '-*.pth')) == 0:
            return

        f_list = glob.glob(os.path.join(args.save_dir) + '-*.pth')
        epoch_list = [int(i.split('-')[-1].split('.')[0]) for i in f_list]
        start_epoch = sorted(epoch_list)[-1]
        name =  '-{}.pth'.format(start_epoch)
        file_path = os.path.join(args.save_dir, name)
        print("loading model", file_path)

        if opt.device == torch.device('cuda'):
            state = torch.load(file_path)
        else:
            state = torch.load(file_path, map_location=opt.device)

        self._epoch = state['epoch']
        self._iter = state['iter']
        self.state_dict =  self.model.state_dict(),
        self.max_acc = state['max_acc']


    def train(self):

        start_time = time.time()
        event_generator = self.dataset.events()
        while True:
            try:
                event = next(event_generator)
                self._iter += 1
            except StopIteration:

                print('This epoch is done.')
                break
            self.optimizer.zero_grad()
            output, loss, result, atten_dist = self.model(event)
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
                # print(event_num)
            except StopIteration:
                print('This epoch is done.')
                print()
                print()
                break
            output, loss, result, attendist= self.model(event)
            print(loss)
            if result >= 0.5:
                result = 1
            else:
                result = 0
            if result == event.label:
                true_num += 1

        self._epoch += 1
        acc = true_num/event_num
        print(acc)
        if self.max_acc is None:
            self.save_model()
            self.max_acc = acc
        elif self.max_acc < acc:
            self.save_model()
            self.max_acc = acc





if __name__ == '__main__':
    preview_args(args)
    train = Train()
    for i in range(0, args.epoch):
        train.train()

    # print(train.dataset.vocab.vocab_size)
    # print(len(train.embedding_matrix))


