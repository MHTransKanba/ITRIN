import copy
import math
import sys

import torch
from torch import nn, reshape
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from modeling_utils import BertSelfEncoder, BertCrossEncoder_AttnMap, BertPooler, BertLayerNorm, BERTGRUSentiment, \
    GatedConnection,GatedFusionNew,SummaryAttn,l2norm
import torch.nn.functional as F

from transformers import RobertaModel, AutoConfig

import logging

logger = logging.getLogger(__name__)


# roberta模型config部分参数及其对应值
# {
#     "attention_probs_dropout_prob": 0.1,
#     "hidden_act": "gelu",
#     "hidden_dropout_prob": 0.1,
#     "hidden_size": 768,
#     "initializer_range": 0.02,
#     "intermediate_size": 3072,
#     "layer_norm_eps": 1e-05,
#     "max_position_embeddings": 514,
#     "model_type": "roberta",
#     "num_attention_heads": 12,
#     "num_hidden_layers": 12,
#     "type_vocab_size": 1,
#     "vocab_size": 50265
# }

class Coarse2Fine(nn.Module):
    def __init__(self, roberta_name="./roberta", img_feat_dim=2048):
        super().__init__()
        self.img_feat_dim = img_feat_dim
        # config 是一个包含了模型配置信息的对象，其类型是 BertConfig 或其子类（因为 RoBERTa 是 BERT 的一种变体）。通过这个配置对象，你可以访问和了解加载模型的各种参数设置。
        # vocab_size hidden_size num_attention_heads num_hidden_layers intermediate_size hidden_dropout_prob...
        config = AutoConfig.from_pretrained(roberta_name)
        self.hidden_dim = config.hidden_size

        self.roberta = RobertaModel.from_pretrained(roberta_name)
        self.sent_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.feat_linear = nn.Linear(self.img_feat_dim, self.hidden_dim)  # [N*n, 100, 2048] ->[N*n, 100, 768]
        self.img_self_attn = BertSelfEncoder(config, layer_num=1)  # [N*n, 100, 768] ->[N*n, 100, 768]

        self.t2v = BertCrossEncoder_AttnMap(config, layer_num=1)
        self.dropout1 = nn.Dropout(0.1)
        self.gather = nn.Linear(self.hidden_dim, 1)
        self.dropout2 = nn.Dropout(0.1)
        self.pred = nn.Linear(128, 2)
        self.ce_loss = nn.CrossEntropyLoss()

        #跨模态对齐+选通
        self.gated_new = GatedFusionNew(768, 4, 0.3)

        #输出层
        self.lin1 = nn.Sequential(
            nn.Linear(1536, 768, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2), )
        self.lin2 = nn.Sequential(
            nn.Linear(1536, 768, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2), )
        #实验得出共享一个门控融合层效果比两个好
        self.gate = GatedConnection(self.hidden_dim, 0.1)
        self.first_pooler = BertPooler(config)
        self.senti_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.senti_detc = nn.Linear(self.hidden_dim, 3)
        self.init_weight()

    # 线性层（nn.Linear）和嵌入层（nn.Embedding）:这些层的权重使用均值为0，标准差为0.02的正态分布进行初始化。这种初始化方法受启发于原始的BERT论文，BERT使用的是截断正态分布来初始化模型的权重。使用小的随机数初始化可以帮助模型在训练初期保持稳定，防止激活值和梯度过大或过小。
    # BertLayerNorm（层归一化）:
    # 层归一化在训练深度神经网络时常用于稳定内部网络状态，通过对输入特征进行归一化，使其均值为0，方差为1。在初始化时，权重（通常是缩放参数）初始化为1，偏置（通常是偏移参数）初始化为0。这样的初始化确保了在训练初期，归一化层不会对网络的表现产生太大影响（即一开始就是一个中性的影响）。
    # 有偏置的线性层:
    # 对于线性层的偏置使用零初始化是常见的做法。零初始化偏置可以确保输出在没有输入的情况下为零，从而不会引入不必要的偏移。这可以在模型训练初期避免偏差带来的不利影响。

    # 这段代码的作用是根据指定的初始化策略对模型中除了roberta模型的不同类型参数进行初始化，确保模型的权重和偏置参数处于合理的初始状态，有利于训练过程的顺利进行。
    def init_weight(self):
        ''' bert init
        '''
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)) and ('roberta' not in name):  # linear/embedding
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, BertLayerNorm) and ('roberta' not in name):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None and ('roberta' not in name):
                module.bias.data.zero_()

    def forward(self, img_id,
                input_ids, input_mask, img_feat,
                relation_label,
                pred_loss_ratio=1.):
        # input_ids,input_mask : [N, L]
        #             img_feat : [N, 100, 2048]
        #         spatial_feat : [N, 100, 5]
        #            box_label : [N, 1, 100 ]
        # box_labels: if IoU > 0.5, IoU_i/sum(); 0

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size, seq = input_ids.size()
        _, roi_num, feat_dim = img_feat.size()  # =100

        # text feature extraction
        # 因此，sentence_output 中的每个元素都是输入句子中每个 token 的表示n*l*768，而 text_pooled_output 是整个句子的池化表示n*768。
        roberta_output = self.roberta(input_ids, input_mask)

        sentence_output = roberta_output.last_hidden_state  # [N, L, 768]
        text_pooled_output = roberta_output.pooler_output  # [30*768]


        # sentence_output = self.sent_dropout(sentence_output)

        # visual self Attention
        img_feat_ = self.feat_linear(img_feat)  # [N, 100, 2048] ->[N, 100, 768]
        image_mask = torch.ones((batch_size, roi_num)).to(device)  # [N, 100]
        extended_image_mask = image_mask.unsqueeze(1).unsqueeze(2)  # [N, 1, 1, 100]
        extended_image_mask = extended_image_mask.to(dtype=next(self.parameters()).dtype)  # float or half
        extended_image_mask = (1.0 - extended_image_mask) * -10000.0  # [N, 1, 1, 100]
        visual_output = self.img_self_attn(img_feat_, extended_image_mask)  # [N, 100, 768]
        visual_output = visual_output[-1]  # [N, 100, 768]

        # 1. sentence query visual:经过这一步结果会于q的维度相同。
        sentence_aware_image, _ = self.t2v(sentence_output,
                                           visual_output,
                                           extended_image_mask,
                                           output_all_encoded_layers=False)  # sentence query image
        sentence_aware_image = sentence_aware_image[-1]  # [N,128,768]

        gathered_sentence_aware_image = self.gather(self.dropout1(
            sentence_aware_image)).squeeze(2)  # [N,128,768]->[N,128,1] ->[N,128]
        rel_pred = self.pred(self.dropout2(
            gathered_sentence_aware_image))  # [N,2]

        gate = torch.softmax(rel_pred, dim=-1)[:, 1].unsqueeze(1).expand(batch_size,
                                                                         128).unsqueeze(2).expand(batch_size,
                                                                                                  128,
                                                                                                  self.hidden_dim)  # [N,1,100]
        gated_sentence_aware_image = gate * sentence_aware_image

        fused_f1,fused_f2 = self.gated_new(gated_sentence_aware_image,sentence_output)

        # co_v1 = self.final_reduce_1(co_v1, co_v1)
        # co_v2 = self.final_reduce_2(co_v2, co_v2)
        # co_v1 = l2norm(co_v1)
        # co_v2 = l2norm(co_v2)

        fused = fused_f1
        h = torch.cat((fused, gated_sentence_aware_image), -1)
        h = self.lin1(h)

        h2 = torch.cat((fused, sentence_output), -1)
        h2 = self.lin2(h2)
        # senti_pred = 0.2*h + 0.8*h2


        res = torch.zeros(batch_size, 128, self.hidden_dim).to(device)
        for i in range(rel_pred.size(0)):
            if rel_pred[i][0] >= rel_pred[i][1]:
                res[i, :, :] = self.gate(h2[i, :, :], h[i, :, :])
            else:
                res[i, :, :] = self.gate(h[i, :, :], h2[i, :, :])
        senti_pred = self.first_pooler(res)
        senti_pred = self.senti_dropout(senti_pred)
        senti_pred = self.senti_detc(senti_pred)
        if relation_label != None:
            pred_loss = self.ce_loss(rel_pred, relation_label.long())
        else:
            pred_loss = torch.tensor(0., requires_grad=True).to(device)



        # senti_pred：三分类最后的情感标签   pred_loss：图片文本关系分类的损失 rel_pred：图片文本关系分类的预测结果 Attn_map：图片文本对齐的注意力分布
        return senti_pred, pred_loss_ratio * pred_loss, rel_pred

