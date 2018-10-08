import torch.nn as nn
import torch.nn.functional as F
import torch
import math


# Attention: K(L, B, H), V(L, B, H), Q(1, B, H) -> c(1, B, H), attn(1, B, L)
# Attention: K(B, h, L, H_k), V(B, h, L, H_k), Q(B, h, 1, H_k) -> c(B, h, 1, H_k), attn(B, h, 1, L)
class Attention(nn.Module):  # 对应的输出为 tuple: (context, attn)
    def __init__(self, args, score_function='scaled_dot_product'):
        super(Attention, self).__init__()
        self.args = args
        self.score_function = score_function

        if self.score_function == "general":
            pass  # 以后用到再写
        elif self.score_function == "concat":
            self.W = nn.Linear(2 * self.args.hidden_dim, 1)
        elif self.score_function == "perceptron":
            self.W_q = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)  # v(tanh(W_q * Q + W_k * K))
            self.W_k = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
            self.v = nn.Linear(self.args.hidden_dim, 1)
            if self.args.coverage:
                self.W_c = nn.Linear(1, self.args.hidden_dim)

    def forward(self, query, key, value, X_padding_mask=None, coverage=None, dropout=0.1):
        """
        K / key: (L, B, H) encoder_outputs, encoder feature
        V / value: (L, B, H) to calculate the context vector
        Q / query: (1, B, H) last_hidden, deocder feature
        X_padding_mask: (B, 1, L)
        coverage: (B, L)
        """
        # query: (1, B, H); 因为对应的 RNN 输入要求 (L, B, H)
        # key 和 value: (L, B, H), 这样将前两个转置后效果就相同了
        # 四维的对应的是 MultiHeadedAttention


        dim = query.size(-1)
        query = query.transpose(0, 1)  # -> (B, 1, H)
        key = key.transpose(0, 1)  # -> (B, L, H)
        value = value.transpose(0, 1)  # -> (B, L, H)
        #(32,1)

        # default for 'Scaled Dot Product Attention'
        if self.score_function == "scaled_dot_product":
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim)  # (B, 1, H) x (B, H, L) -> (B, 1, L)

        attn_dist = F.softmax(scores, dim=-1)  # (B, 1, L)
        attn_dist = F.dropout(attn_dist, p=dropout)
        context = torch.matmul(attn_dist, value).transpose(0, 1)  # (B, 1, L) x (B, L, H) -> (B, 1, H) -> (1, B, H)

        return context, attn_dist.transpose(0, 1)  # (1, B, H) 的语义向量和 (1, B, L) 的当前 decoder h_i 的权重矩阵;
if __name__ == '__main__':
    h = 5   # 切片个数, MultiHeadedAttention 中的参数
    L = 10  # sen_len
    B = 3   # batch_size
    H = 20  # hidden_dim

    # 测试 Attention
    query = torch.rand(1, B, H)  # 1, 3, 20
    key = torch.rand(L, B, H)  # 10, 3, 20
    value = key  # 10, 3, 20
    coverage = torch.rand(B, L)
    model = Attention()
    print(model(query, key, value)[0].shape)  # torch.Size([1, 3, 20])
    print(model(query, key, value)[1].shape)  # torch.Size([1, 3, 10])
    model = Attention(hidden_dim=H, score_function="concat")
    print(model(query, key, value)[0].shape)  # torch.Size([1, 3, 20])
    print(model(query, key, value)[1].shape)  # torch.Size([1, 3, 10])
    model = Attention(hidden_dim=H, score_function="perceptron")
    print(model(query, key, value, coverage=coverage)[0].shape)  # torch.Size([1, 3, 20])
    print(model(query, key, value, coverage=coverage)[1].shape)  # torch.Size([1, 3, 10])

    # 测试 h, L, B, H for MultiHeadedAttention
    query = torch.rand(B, h, 1, H)  # 3, 5, 1, 20
    key = torch.rand(B, h, L, H)  # 3, 5, 10, 20
    value = key  # 10, 3, 20
    coverage = torch.rand(B, L)
    model = Attention()
    print(model(query, key, value)[0].shape)  # torch.Size([3, 5, 1, 20])
    print(model(query, key, value)[1].shape)  # torch.Size([3, 5, 1, 10])
    model = Attention(hidden_dim=H, score_function="concat")
    print(model(query, key, value)[0].shape)  # torch.size([3, 5, 1, 20])
    print(model(query, key, value)[1].shape)  # torch.Size([3, 5, 1, 10])
    model = Attention(hidden_dim=H, score_function="perceptron")
    print(model(query, key, value, coverage=coverage)[0].shape)  # torch.Size([3, 5, 1, 20])
    print(model(query, key, value, coverage=coverage)[1].shape)  # torch.Size([3, 5, 1, 10])

    # test X_padding_mask
    X_padding_mask = torch.rand(B, 1, L)
    model = Attention()
    print(model(query, key, value, X_padding_mask)[0].shape)  # torch.Size([3, 5, 1, 20])
    print(model(query, key, value, X_padding_mask)[1].shape)  # torch.Size([3, 5, 1, 10])

    # 测试矩阵相乘
    # 测试 4 维输入对应的 Attention
    a = torch.rand(B, 1, H)  # 3, 1, 20
    b = torch.rand(B, H, L)  # 3, 20, 10
    print(torch.matmul(a, b).shape)  # torch.Size([3, 1, 10])
    a = torch.rand(h, B, 1, H)  # 5, 3, 1, 20
    b = torch.rand(h, B, H, L)  # 5, 3, 20, 10
    print(torch.matmul(a, b).shape)  # torch.Size([5, 3, 1, 10])


