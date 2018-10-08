import json

from utils.content import Content


class Event:
    def __init__(self, args, file, label=0, vocab=None):
        self.args = args
        self.vocab = vocab
        self.id = file.split('.')[0].split('/')[-1]
        self.label = label
        # self.all_content = list(Content(json.load(line) for line in open(file)))
        self.original_content = Content(self.args, json.load(open(file))[0], vocab)
        self.reply_contents = list(Content(self.args, dic, vocab, is_reply=True) for dic in json.loads(open(file).read())[1:10000])
        # self.data = vocab.sen2id(self.original_content.tokens)+ vocab.sen2id(self.reply.tokens)
        self.len_reply = len(self.reply_contents)
        self.sentence = list(content.tokens for content in [self.original_content] + self.reply_contents)


if __name__ == '__main__':
    with open('/Users/gengyue/Desktop/rumdect/Weibo/4010312877.json') as f:
        dic = json.load(f)
        # pprint(dic)
    event = Event('/Users/gengyue/datasets/Rumor/rumdect/Weibo/3908462705102897.json')
    print(event.id)
    print(event.original_content.original_text)
    # dic = json.loads('/Users/gengyue/Desktop/rumdect/Weibo/4010312877.json')





