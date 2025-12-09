import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    #生成基于正弦函数的位置编码

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats      #位置编码特征数量
        self.temperature = temperature          #用于调整位置编码的尺度
        self.normalize = normalize              #是否对位置编码进行归一化
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, class_token=False):
        #正弦位置编码的计算
        #x的形状（B，N，C），B：批次大小，N：序列长度，C：特征数量
        x = x.permute(1, 2, 0)

        num_feats = x.shape[1]
        num_pos_feats = num_feats       #位置编码的特征数量与输入特征数量相同
        mask = torch.zeros(x.shape[0], x.shape[2], device=x.device).to(torch.bool)      #用于后续计算序列中每个位置的累积和
        batch = mask.shape[0]
        assert mask is not None     #确保mask不是None，断言检查
        not_mask = ~mask            #取mask的逻辑非，得到一个形状为(N, B)的布尔张量，其中非零位置为True
        y_embed = not_mask.cumsum(1, dtype=torch.float32)       #沿着第二维度（批次维度）累加1（True位置），得到每个位置的累积位置索引

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)       #创建一个从0到num_pos_feats-1的整数序列，表示每个位置编码的特征索引
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)                  #计算每个位置编码特征的调整因子

        pos_y = y_embed[:, :, None] / dim_t
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        return pos_y


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    #设置激活函数
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    #后归一化和前归一化是等价的，在实际应用中可能会因为数值稳定性和训练动态的不同而有所差异
    #残差连接、层归一化、前馈网络作用（后处理步骤，后归一化）
    def forward_post(self,
                     src,
                     src_mask = None,
                     src_key_padding_mask = None,
                     pos = None):
        src2 = self.self_attn(src, src, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    #残差连接、层归一化、前馈网络作用（前处理步骤，预归一化）
    def forward_pre(self, src,
                    src_mask = None,
                    src_key_padding_mask = None,
                    pos = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    #判断使用预归一化还是后归一化（判断参数：normalize_before）
    def forward(self, src,
                src_mask = None,
                src_key_padding_mask = None,
                pos = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class History_motion_embedding(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=512, dropout=0.1,
                 activation='relu', normalize_before=False, pos_type='sin'):
        super(History_motion_embedding, self).__init__()
        self.cascade_num = 6        #enconder层数
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))       #设置可学习的token（表示序列的开始），用于嵌入
        self.trca = nn.ModuleList()     #用于存储Module,这里是encoder layer
        for _ in range(self.cascade_num):
            self.trca.append(TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                   dropout, activation, normalize_before))

        self.proj = nn.Linear(8, d_model)       #创建了一个线性层（全连接层），输入维度:8,输出维度：d_model，高维映射
        if pos_type == 'sin':
            self.pose_encoding = PositionEmbeddingSine(normalize=True)

    #输出embedding
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0).to(self.cls_token)
        else:
            x = x.to(self.cls_token)

        q_patch = self.proj(x).permute(1, 0, 2)
        pos = self.pose_encoding(q_patch).transpose(0, 1)
        n, b, d = q_patch.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b).permute(1, 0, 2).contiguous()
        encoder_patch = torch.cat((q_patch, cls_tokens), dim=0)

        for i in range(self.cascade_num):
            en_out = self.trca[i](src=encoder_patch, pos=pos)
            encoder_patch = en_out

        out = en_out[0].view(b, 1, d).contiguous()
        return out

