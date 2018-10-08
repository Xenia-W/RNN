from torch import nn
import torch
from layers.attention import Attention

from config import opt


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.embed = nn.Embedding(self.args.vocab_size, self.args.embed_dim).to(opt.device)
        self.gru = nn.GRU(300, self.args.hidden_dim).to(opt.device)
        self.bi_gru = nn.GRU(300, self.args.hidden_dim, bidirectional=True).to(opt.device)
        self.bi_gru_content = nn.GRU(300, self.args.hidden_dim, bidirectional=True).to(opt.device)
        self.bi_gru_reply = nn.GRU(300, self.args.hidden_dim, bidirectional=True).to(opt.device)
        self.bi_gru_user_profile = nn.GRU(11, self.args.hidden_dim, bidirectional=True).to(opt.device)
        self.output_linear_3 = nn.Linear(3 * self.args.batch_size * self.args.hidden_dim, 1).to(opt.device)
        self.output_linear_2 = nn.Linear(2 * self.args.batch_size * self.args.hidden_dim, 1).to(opt.device)
        self.output_linear_1 = nn.Linear(1 * self.args.batch_size * self.args.hidden_dim, 1).to(opt.device)

        self.linear = nn.Linear(self.args.hidden_dim*2, self.args.hidden_dim).to(opt.device) # output: (L, B, 2*H) -> (L, B, H)
        self.hidden_linear = nn.Linear(1*self.args.batch_size*self.args.hidden_dim,1*self.args.batch_size*self.args.hidden_dim).to(opt.device)
        self.attention = Attention(self.args)

    def forward(self, event):
        original_content = torch.sum(self.embed(torch.tensor(event.original_content.data()).to(opt.device)).to(opt.device),dim=0).to(opt.device)
        reply_content = torch.stack([torch.sum(self.embed(torch.tensor(reply_content.data()).to(opt.device)).to(opt.device),dim=0).to(opt.device)for reply_content in event.reply_contents]).to(opt.device)
        # original_content_data = torch.sum(self.embed(torch.tensor(event.original_content.data()).to(opt.device)).to(opt.device), dim=0).to(opt.device)
        # original_content_feature = torch.tensor(event.original_content.feature(), dtype=torch.float).to(opt.device)
        # original_content = torch.cat((original_content_data, original_content_feature)).to(opt.device)
        # reply_content = torch.stack([torch.cat((torch.sum(self.embed(torch.tensor(reply_content.data()).to(opt.device)).to(opt.device), dim=0), torch.tensor(reply_content.feature(), dtype=torch.float).to(opt.device))).to(opt.device) for reply_content in event.reply_contents]).to(opt.device)
        # 原文加评论

        # X_data = torch.cat((original_content.unsqueeze(0), reply_content)).to(opt.device)  # (L, D)
        # X_data = X_data.unsqueeze(1).to(opt.device)  # (L, D) -> (L, B, D)/(L, 1, D)
        # output, hidden = self.bi_gru(X_data)  # (L, B, 2*H), (2, B, H)
        # output = self.linear(output)
        # hidden = torch.sum(hidden, dim=0).unsqueeze(0)
        # hidden, atten_dist = self.attention(hidden, output, output)
        # hidden = hidden.flatten()  # (2, B, H) -> (2*B*H)

        # 原文
        original_content = torch.sum(self.embed(torch.tensor(event.original_content.data()).to(opt.device)).to(opt.device),dim=0).to(opt.device)
        X_ori_content_data =(original_content.unsqueeze(0)).to(opt.device)
        X_ori_content_data = X_ori_content_data.unsqueeze(1).to(opt.device)
        output_ori, hidden_ori = self.bi_gru_content(X_ori_content_data)

        output_ori = self.linear(output_ori)
        hidden_ori = torch.sum(hidden_ori, dim=0).unsqueeze(0)
        hidden_ori, atten_dist_ori = self.attention(hidden_ori, output_ori, output_ori)
        hidden_ori = hidden_ori.flatten()
        multi_view = hidden_ori.to(opt.device)

        # 评论
        if self.args.reply:
            # print("Using reply")
            reply_content = torch.stack([torch.sum(self.embed(torch.tensor(reply_content.data()).to(opt.device)).to(opt.device),dim=0).to(opt.device)for reply_content in event.reply_contents]).to(opt.device)
            X_reply_content_data = reply_content.unsqueeze(1).to(opt.device)
            output_reply, hidden_reply = self.bi_gru_reply(X_reply_content_data)

            output_reply = self.linear(output_reply)
            hidden_reply = torch.sum(hidden_reply, dim=0).unsqueeze(0)
            hidden_reply, atten_dist_reply = self.attention(hidden_reply, output_reply, output_reply)
            hidden_reply = hidden_reply.flatten()
            multi_view = torch.cat((multi_view, hidden_reply)).to(opt.device)

        # user profile
        if self.args.profile:
            # print("Using profile")
            original_profile = torch.tensor(event.original_content.feature(), dtype=torch.float).to(opt.device)
            reply_profile = torch.stack([(torch.tensor(reply_content.feature(), dtype=torch.float).to(opt.device)).to(opt.device) for reply_content in event.reply_contents]).to(opt.device)
            X_profile_data = torch.cat((original_profile.unsqueeze(0), reply_profile)).to(opt.device)
            X_profile_data = X_profile_data.unsqueeze(1).to(opt.device)
            output_profile,hidden_profile = self.bi_gru_user_profile(X_profile_data)

            output_profile = self.linear(output_profile)
            hidden_profile = torch.sum(hidden_profile, dim=0).unsqueeze(0)
            hidden_profile, atten_dist_profile = self.attention(hidden_profile, output_profile, output_profile)
            hidden_profile = hidden_profile.flatten()
            multi_view = torch.cat((multi_view, hidden_profile)).to(opt.device)


        # output: (L, B, 2*H) -> (L, B, H)
        # hidden: (2, B, H) -> (1, B, H)
        if self.args.profile:
            result = torch.sigmoid(self.output_linear_3(multi_view))  # (2*B*H) -> n
        elif self.args.reply:
            result = torch.sigmoid(self.output_linear_2(multi_view))
        else:
            result = torch.sigmoid(self.output_linear_1(multi_view))


        loss = (event.label - result)**2

        return output_ori, loss, result, atten_dist_ori


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
