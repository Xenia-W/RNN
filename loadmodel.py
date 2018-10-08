from train import Train

class Loadmodel:
    def __init__(self):
        self.train = Train()

        print(self.train.max_acc)


if __name__ == '__main__':
    l = Loadmodel()

