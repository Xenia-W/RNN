import re
import jieba
import numpy as np

from config import opt


class Content:
    def __init__(self, args, content, vocab, is_reply=False):
        """
        :param content: dict
        """
        self.args = args
        self.original_text = content['original_text']
        self.tokens = list(jieba.cut(self.original_text, cut_all=False))
        self.t = content['t']
        self.user_description = list(jieba.cut(content['user_description'],cut_all = False))
        self.friends_count = content['friends_count']
        self.followers_count = content['followers_count']
        self.verified_type = content['verified_type']
        self.verified_reason = list(jieba.cut(content['verified_reason'],cut_all = False))
        self.favourites_count = content['favourites_count']
        self.user_geo_enabled = 1 if content['user_geo_enabled'] else 0
        self.bi_followers_count = content['bi_followers_count']
        self.city = int(content['city'])
        self.province = int(content['province'])

        if is_reply:
            self.original_text_data = self.pad(vocab.sen2id(self.tokens), opt.max_document_len)
        else:
            self.original_text_data = self.pad(vocab.sen2id(self.tokens), opt.max_document_len)
        self.user_descri_data = self.pad(vocab.sen2id(self.user_description), opt.max_user_description_len)
        self.veri_reason_data = self.pad(vocab.sen2id(self.verified_reason), opt.max_verified_reason_len)

    def num_url(self):
        url = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',self.original_text )
        return len(url)

    def has_at(self):
        return self.original_text.find(' @')

    def data(self):
        return self.original_text_data + self.user_descri_data + self.veri_reason_data

    def feature(self):
        return np.array([self.t] + [self.bi_followers_count] +
                        [self.city] + [self.favourites_count] + [self.followers_count] +
                        [self.friends_count] + [self.has_at()] + [self.num_url()] +
                        [self.verified_type] +
                        [self.user_geo_enabled] + [self.province])

    @staticmethod
    def pad(sen_ids, sen_len):
        if len(sen_ids) < sen_len:
            sen_ids = sen_ids + [opt.pad_idx] * (sen_len - len(sen_ids))
        else:
            sen_ids = sen_ids[:sen_len]
        return sen_ids



if __name__ == '__main__':
    from pprint import pprint
    import json
    f = open('/Users/gengyue/Desktop/rumdect/Weibo/4010312877.json')
    content = json.load(f)[0]
    content = Content(content)
    # content = dic[0]
    #content = Content(dic[0])

    # dic = json.loads('/Users/gengyue/Desktop/rumdect/Weibo/4010312877.json')

    print(content.num_url())
    print(content.has_at())
    print(content.original_text)
    print(content.tokens)
    print(content.t)
    print(content.user_description)
    print(content.friends_count)
    print(content.followers_count)
    print(content.verified_type)
    print(content.verified_reason)
    print(content.favourites_count)
    print(content.user_geo_enabled)
    print(content.bi_followers_count)
    print(content.city)
    print(content.province)

