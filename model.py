from torch import nn
import torch
from attention import Attention


import config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim).cuda()
        self.gru = nn.GRU(300, config.hidden_dim).cuda()
        self.bi_gru = nn.GRU(300, config.hidden_dim, bidirectional=True).cuda()
        self.output_linear = nn.Linear(1 * config.batch_size * config.hidden_dim, 1).cuda()
        self.linear = nn.Linear(config.hidden_dim*2,config.hidden_dim).cuda() # output: (L, B, 2*H) -> (L, B, H)
        self.hidden_linear = nn.Linear(1*config.batch_size*config.hidden_dim,1*config.batch_size*config.hidden_dim).cuda()
        self.attention = Attention()


    def forward(self, event):
        original_content = torch.sum(self.embed(torch.tensor(event.original_content.data()).cuda()).cuda(),dim=0).cuda()
        reply_content = torch.stack([torch.sum(self.embed(torch.tensor(reply_content.data()).cuda()).cuda(),dim=0).cuda()for reply_content in event.reply_contents]).cuda()
        original_content_data = torch.sum(self.embed(torch.tensor(event.original_content.data()).cuda()).cuda(), dim=0).cuda()
        # original_content_feature = torch.tensor(event.original_content.feature(), dtype=torch.float).cuda()
        #original_content = torch.cat((original_content_data, original_content_feature)).cuda()
        # reply_content = torch.stack([torch.cat((torch.sum(self.embed(torch.tensor(reply_content.data()).cuda()).cuda(), dim=0), torch.tensor(reply_content.feature(), dtype=torch.float).cuda())).cuda() for reply_content in event.reply_contents]).cuda()

        X_data = torch.cat((original_content.unsqueeze(0), reply_content)).cuda()  # (L, D)
        X_data = X_data.unsqueeze(1).cuda()  # (L, D) -> (L, B, D)/(L, 1, D)
        output, hidden = self.bi_gru(X_data)  # (L, B, 2*H), (2, B, H)

        output = self.linear(output)
        # output: (L, B, 2*H) -> (L, B, H)
        # hidden: (2, B, H) -> (1, B, H)

        hidden = torch.sum(hidden, dim=0)
        hidden = hidden.unsqueeze(0)

        hidden, atten_dist = self.attention(hidden, output, output)


        hidden = hidden.flatten()  # (2, B, H) -> (2*B*H)
        result = torch.sigmoid(self.output_linear(hidden))  # (2*B*H) -> n

        loss = (event.label - result)**2

        return output, loss, result

if __name__ == '__main__':
    model = Model()
    batch = torch.tensor([[0, 1]])
    print(batch.shape)
    print(model.embed(batch).shape)


# ## GRU(input_size, hidden_size, num_layers)
# gru = nn.GRU(10, 20, bidirectional=True)
# input = torch.randn(5, 3, 10)  # 每个句子 5 个单词, 输入 3 个句子, 每个单词 10 维词向量; L, B, D
# output, hn = gru(input); print(output.shape, hn.shape)  # torch.Size([5, 3, 20]) torch.Size([1, 3, 20])
# print(output)
# print(hn)  # 可以看到, hn 和 output 的最后一个是一样的...
