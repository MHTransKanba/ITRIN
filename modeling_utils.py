from torch import nn
from math import sqrt
import torch
import math
import copy
import sys
import numpy as np
import logging
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torchvision import transforms as tfs
logger = logging.getLogger(__name__)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


#L2 归一化的主要作用是使得数据更具有可比性和稳定性，而不是让特征更满足某一特定的分布。用于解决特征的尺度不一致问题、抑制异常值的影响、梯度下降的稳定性等问题。
def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def qkv_attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(d_k)
    # if mask is not None:
    #     scores.data.masked_fill_(mask.data.eq(0), -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

def sum_attention(nnet, query, value, dropout=None):
    scores = nnet(query).transpose(-2, -1)
    # if mask is not None:
    #     scores.data.masked_fill_(mask.data.eq(0), -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
class SummaryAttn(nn.Module):
    def __init__(self, dim, num_attn, dropout, is_cat=False):
        super(SummaryAttn, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, num_attn),
        )
        self.h = num_attn
        self.is_cat = is_cat
        self.attn = None
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, query, value):
        # if mask is not None:
        #     mask = mask.unsqueeze(-2)
        batch = query.size(0)

        weighted, self.attn = sum_attention(self.linear, query, value, dropout=self.dropout)
        weighted = weighted if self.is_cat else weighted.mean(dim=-2)

        return weighted


#跨模态对齐和修正模块
class GatedFusionNew(nn.Module):
    def __init__(self, dim, num_attn, dropout=0.01, reduce_func="self_attn", fusion_func="concat"):
        super(GatedFusionNew, self).__init__()
        self.dim = dim
        self.h = num_attn

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

        self.reduce_func = reduce_func
        self.fusion_func = fusion_func

        self.img_key_fc = nn.Linear(dim, dim, bias=False)
        self.txt_key_fc = nn.Linear(dim, dim, bias=False)

        self.img_query_fc = nn.Linear(dim, dim, bias=False)
        self.txt_query_fc = nn.Linear(dim, dim, bias=False)

        self.weighted_img_key_fc = nn.Linear(dim, dim, bias=False)
        self.weighted_txt_key_fc = nn.Linear(dim, dim, bias=False)

        self.weighted_img_query_fc = nn.Linear(dim, dim, bias=False)
        self.weighted_txt_query_fc = nn.Linear(dim, dim, bias=False)

        in_dim = dim
        if fusion_func == "sum":
            in_dim = dim
        elif fusion_func == "concat":
            in_dim = 2 * dim
        else:
            raise NotImplementedError('Only support sum or concat fusion')

        self.img_linear=nn.Linear(768,768)
        self.img_txt_linear=nn.Linear(768,768)
        self.att_linear=nn.Linear(768*2,1)

        self.fc_1 = nn.Sequential(
            nn.Linear(in_dim, dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout), )
        self.fc_2 = nn.Sequential(
            nn.Linear(in_dim, dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout), )
        self.fc_out = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

        self.init_weights()
        print("MHGatedFusion module init success!")

    def init_weights(self):
        r = np.sqrt(6.) / np.sqrt(self.dim +
                                  self.dim)
        self.img_key_fc.weight.data.uniform_(-r, r)
        self.txt_key_fc.weight.data.uniform_(-r, r)
        self.fc_1[0].weight.data.uniform_(-r, r)
        # self.fc_1[0].bias.data.fill_(0)
        self.fc_2[0].weight.data.uniform_(-r, r)
        # self.fc_2[0].bias.data.fill_(0)
        self.fc_out[0].weight.data.uniform_(-r, r)
        self.fc_out[0].bias.data.fill_(0)
        self.fc_out[3].weight.data.uniform_(-r, r)
        self.fc_out[3].bias.data.fill_(0)

    def forward(self, v1, v2):
        # 跨模态对齐机制
        k1 = self.img_key_fc(v1)
        k2 = self.txt_key_fc(v2)
        q1 = self.img_query_fc(v1)
        q2 = self.txt_query_fc(v2)


        weighted_v1, attn_1 = qkv_attention(q2, k1, v1)

        weighted_v2, attn_2 = qkv_attention(q1, k2, v2)

        weighted_v2_q = self.weighted_txt_query_fc(weighted_v2)
        weighted_v2_k = self.weighted_txt_key_fc(weighted_v2)

        weighted_v1_q = self.weighted_img_query_fc(weighted_v1)
        weighted_v1_k = self.weighted_img_key_fc(weighted_v1)

        fused_v1, _ = qkv_attention(weighted_v2_q, weighted_v2_k, weighted_v2)

        fused_v2, _ = qkv_attention(weighted_v1_q, weighted_v1_k, weighted_v1)

        fused_v1 = l2norm(fused_v1)
        fused_v2 = l2norm(fused_v2)

        # 跨模态修正机制
        gate_v1 = F.sigmoid((v1 * fused_v1).sum(dim=-1)).unsqueeze(-1)

        gate_v2 = F.sigmoid((v2 * fused_v2).sum(dim=-1)).unsqueeze(-1)

        if self.fusion_func == "sum":
            co_v1 = (v1 + fused_v1) * gate_v1
            co_v2 = (v2 + fused_v2) * gate_v2
        elif self.fusion_func == "concat":
            co_v1 = torch.cat((v1, fused_v1), dim=-1) * gate_v1
            co_v2 = torch.cat((v2, fused_v2), dim=-1) * gate_v2
        covtt = self.fc_1(co_v1)
        co_v1 = covtt + v1
        co_v2 = self.fc_2(co_v2) + v2

        return co_v1, co_v2
def Linear(inputdim, outputdim, bias=True, uniform=True):
    linear = nn.Linear(inputdim, outputdim, bias)
    if uniform:
        nn.init.xavier_uniform_(linear.weight)
    else:
        nn.init.xavier_normal_(linear.weight)
    if bias:
        nn.init.constant_(linear.bias, 0.0)
    return linear

class GatedConnection(nn.Module):
    def __init__(self, size, dropout):
        super(GatedConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.gate = Linear(size*2, 1)
        self.norm = nn.LayerNorm(size)
    def forward(self, x, y):
        #y = sublayer(x)
        y = self.dropout(y)
        g = torch.sigmoid(self.gate(torch.cat((x, y), -1)))
        return g * y + x

class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_layers,
                 bidirectional,
                 dropout):

        super().__init__()
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        # text = [batch size, sent len]

        # embedded = [batch size, sent len, emb dim]
        outs, hidden = self.rnn(text)
        #outs的维度最后与hidden_dim相同
        outs = (outs[:, :, :outs.size(2) // 2] + outs[:, :, outs.size(2) // 2:]) / 2
        o = torch.mean(outs, dim=1)
        return outs, o

class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


## Cross
class BertCrossEncoder_AttnMap(nn.Module):
    def __init__(self, config, layer_num):
        super(BertCrossEncoder_AttnMap, self).__init__()
        layer = BertCrossAttentionLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_attn_maps=[]
        for layer_module in self.layer:
            s1_hidden_states,attn_map = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(s1_hidden_states)
                all_attn_maps.append(attn_map)
        if not output_all_encoded_layers:
            all_encoder_layers.append(s1_hidden_states)
            all_attn_maps.append(attn_map)
        return all_encoder_layers,all_attn_maps

class BertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.attention = BertCrossAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output,attn_map = self.attention(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output,attn_map

class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        self.self = BertCoAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output,attn_map = self.self(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output,attn_map

class BertCoAttention(nn.Module):
    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        mixed_query_layer = self.query(s1_hidden_states)   
        mixed_key_layer = self.key(s2_hidden_states)     
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)  
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))   #[N,12,1,100]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask  

        attn_map=torch.mean(attention_scores,dim=1) #! #[N,12,1,100]->[N,  1, 100]
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attn_map


## Self
class BertSelfEncoder(nn.Module):
    def __init__(self, config, layer_num):
        super(BertSelfEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range( layer_num)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  #[batch_size,seq_len,num_heads,head_size]
        return x.permute(0, 2, 1, 3)   #[batch_size,num_heads,seq_len,head_size]

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # 此处attention_mask句长内为0，外为-10000.[batch_size,1,1,seq_len]
        #mask矩阵为batchsize*1*1*length,qkv矩阵为batchsize*head_size*length*(768/head_size)
        #q*k = batchsize*head_size*length*length,mask利用广播机制可以对注意力矩阵进行遮盖
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) #[batch_size,num_heads,sqe_len,head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] #[batch_size,hidden_size]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



